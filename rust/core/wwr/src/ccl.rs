//! wwr's `src/extension/ccl` (module `wwr.extension.ccl`): the
//! collectives' status as an error code, the communicator, and the typed
//! collectives through it, NCCL on CUDA and RCCL on HIP.
//!
//! | C++ (wwr) | Here |
//! |---|---|
//! | `:ccl_error`, the `wwrcclResult_t` specializations | `impl ErrorCode for wwr_ccl_result` |
//! | `:ccl_comm`, `CommWrapper<P_create, P_destroy>` | [`Comm`], one rank of a clique [`Comm::init_all`] makes |
//! | `CommWrapper`'s single-device and rank constructors | nothing: one process holds every rank, so the clique is made whole |
//! | `wwrcclBroadcast`, `wwrcclBcast` | [`broadcast_send`] (the root, in place, so it only reads) and [`broadcast_recv`] |
//! | `wwrcclReduce` | [`reduce_send`] (a rank that is not the root) and [`reduce_recv`] (the root), a sum |
//! | `wwrcclAllReduce`, `wwrcclAllGather` | [`all_reduce`], a sum, and [`all_gather`] |
//! | `wwrcclSend`, `wwrcclRecv` | [`send`] and [`recv`] |
//! | `wwrcclGroupStart` / `wwrcclGroupEnd` | [`group`], which always ends what it starts |
//!
//! **One process, every rank.** A run is one process holding all its
//! devices (`docs/programming_model.md`, "The run's devices"), so its
//! clique is made at once over the device list, rank `r` on `devices[r]`,
//! and each rank's [`Comm`] is used from the one thread that drives its
//! device: a GPU worker's, in a run. Each call enqueues on a stream of the
//! rank's device and returns; the transfer happens on the devices, every
//! rank's kernel meeting the others', so a collective completes only when
//! every rank has issued it. Two rules follow, both the caller's:
//!
//! - **Every rank issues the same collectives, in the same order**, each
//!   with the same element count and root. A rank that issues another
//!   first, or never issues one, leaves the others' kernels waiting on the
//!   device for good.
//! - **One thread issuing for several ranks wraps them in a [`group`]**: a
//!   rank's call may wait for its peers' before it returns, and only a
//!   group defers every call to its end. A test that drives a clique from
//!   one thread does this; the executor's workers, a thread per rank, need
//!   not.
//!
//! **Safe for the same reason [`blas::gemm`](crate::blas::gemm) is**: each
//! buffer is a [`DeviceSlice`] carrying its length, the element count of a
//! call is its slices' length, and where two must agree an always-on
//! contract holds them. Each rank's buffers are its own, read and written
//! on its own stream, and a buffer may be reused once the collective has
//! completed on that stream, which the stream's order and the executor's
//! events already say. What no rank can check is a peer's count; a peer
//! whose count differs is the first rule broken.
//!
//! **f64 alone.** Every row a collective moves in this tree is `f64`; a
//! caller that needs another type is the reason to add a trait over the
//! C ABI's data types.

use std::panic::Location;

use wwr_sys::{wwr_ccl_comm, wwr_ccl_data_type, wwr_ccl_red_op, wwr_ccl_result};

use crate::common::{ErrorCode, SourceLocation};
use crate::device_slice::{DeviceSlice, DeviceSliceMut};
use crate::handle::{DeviceHandle, DeviceHandleStream};
use crate::runtime::Stream;
use crate::{gpu_log_error_at, gpu_status, gpu_status_at};
use contracts::{debug_requires, requires};
use fb_core_error_handling::{Result, Status};
use fb_core_log::{eq, fits_i32, lt};

/// `wwrcclResult_t`, as the C ABI hands it back. The name is the
/// enumerator's.
impl ErrorCode for wwr_ccl_result {
    const SUCCESS: Self = wwr_ccl_result::WWR_CCL_RESULT_SUCCESS;

    fn name(self) -> &'static str {
        use wwr_ccl_result::*;
        match self {
            WWR_CCL_RESULT_SUCCESS => "WWR_CCL_RESULT_SUCCESS",
            WWR_CCL_RESULT_UNHANDLED_CUDA_ERROR => "WWR_CCL_RESULT_UNHANDLED_CUDA_ERROR",
            WWR_CCL_RESULT_SYSTEM_ERROR => "WWR_CCL_RESULT_SYSTEM_ERROR",
            WWR_CCL_RESULT_INTERNAL_ERROR => "WWR_CCL_RESULT_INTERNAL_ERROR",
            WWR_CCL_RESULT_INVALID_ARGUMENT => "WWR_CCL_RESULT_INVALID_ARGUMENT",
            WWR_CCL_RESULT_INVALID_USAGE => "WWR_CCL_RESULT_INVALID_USAGE",
            WWR_CCL_RESULT_REMOTE_ERROR => "WWR_CCL_RESULT_REMOTE_ERROR",
            WWR_CCL_RESULT_IN_PROGRESS => "WWR_CCL_RESULT_IN_PROGRESS",
        }
    }

    fn description(self) -> &'static str {
        use wwr_ccl_result::*;
        match self {
            WWR_CCL_RESULT_SUCCESS => "no error",
            WWR_CCL_RESULT_UNHANDLED_CUDA_ERROR => "a call to the GPU runtime failed",
            WWR_CCL_RESULT_SYSTEM_ERROR => "a call to the system failed",
            WWR_CCL_RESULT_INTERNAL_ERROR => "an internal check failed",
            WWR_CCL_RESULT_INVALID_ARGUMENT => "an invalid value for an argument",
            WWR_CCL_RESULT_INVALID_USAGE => "the caller used the library wrongly",
            WWR_CCL_RESULT_REMOTE_ERROR => "a remote process failed",
            WWR_CCL_RESULT_IN_PROGRESS => "a non-blocking operation is still in progress",
        }
    }
}

/// One rank of a clique of communicators: its handle, its rank, the
/// clique's size and its device. Made only by [`Comm::init_all`], with
/// every other rank of its clique, and destroyed on drop.
///
/// Neither `Send` nor `Sync`: a rank is used from the thread that drives
/// its device, and the raw handle inside it is not shared.
///
/// ```compile_fail,E0277
/// fn shared_across_threads<T: Send>() {}
/// shared_across_threads::<fb_core_wwr::ccl::Comm>();
/// ```
pub struct Comm {
    raw: wwr_ccl_comm,
    rank: usize,
    ranks: usize,
    dev_idx: i32,
    location: SourceLocation,
}

impl Comm {
    /// A clique over `devices`: rank `r` on `devices[r]`, every rank's
    /// [`Comm`] in rank order, `wwrcclCommInitAll`. The devices are
    /// distinct; the library refuses a device listed twice.
    ///
    /// # Errors
    /// The library's refusal, as a `GpuRuntime` error naming the status:
    /// a device listed twice or not visible, or a clique the devices cannot
    /// form.
    #[requires(lt!(0, devices.len()) && fits_i32!(devices.len()), "a clique has a rank, counted in an i32")]
    #[track_caller]
    pub fn init_all(devices: &[i32]) -> Result<Vec<Comm>> {
        let location = Location::caller();
        let n = devices.len();
        let mut raw = vec![std::ptr::null_mut(); n];
        // `n` fits an i32: the contract.
        let ndev = i32::try_from(n).unwrap_or(i32::MAX);
        // SAFETY: `raw` has room for `ndev` handles and `devices` holds
        // `ndev` device indices, both live locals or borrows for the call.
        let status =
            unsafe { wwr_sys::wwr_ccl_comm_init_all(raw.as_mut_ptr(), ndev, devices.as_ptr()) };
        gpu_status_at(status, location)?;
        Ok(raw
            .into_iter()
            .zip(devices)
            .enumerate()
            .map(|(rank, (raw, &dev_idx))| Comm {
                raw,
                rank,
                ranks: n,
                dev_idx,
                location,
            })
            .collect())
    }

    /// Its rank in the clique, from 0.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The clique's size.
    pub fn ranks(&self) -> usize {
        self.ranks
    }

    /// The raw handle, for a C ABI call.
    pub fn get(&self) -> wwr_ccl_comm {
        self.raw
    }
}

impl DeviceHandle for Comm {
    fn dev_idx(&self) -> i32 {
        self.dev_idx
    }
}

impl Drop for Comm {
    /// Destroy the rank, `wwrcclCommDestroy`; a failure is logged at the
    /// place the clique was made, never raised (`docs/diagnostics.md`,
    /// "`Drop` cannot fail"). Every rank of a clique is destroyed this
    /// way, one at a time, from any one thread.
    fn drop(&mut self) {
        // SAFETY: a live handle from `init_all`, destroyed once.
        let status = unsafe { wwr_sys::wwr_ccl_comm_destroy(self.raw) };
        gpu_log_error_at(status, self.location);
    }
}

/// Run `f`, its collectives deferred to the group's end and issued
/// together, `wwrcclGroupStart` before it and `wwrcclGroupEnd` after: how
/// one thread issues a call for several ranks, and how a rank issues
/// several point-to-point calls, a receive from every peer, as one. The
/// group is ended whatever `f` returns.
///
/// # Errors
/// The first failure: the group's start, `f`'s, or its end.
#[track_caller]
pub fn group(f: impl FnOnce() -> Status) -> Status {
    // SAFETY: a plain library call.
    gpu_status(unsafe { wwr_sys::wwr_ccl_group_start() })?;
    let result = f();
    // SAFETY: as above, ending the group started here.
    let ended = gpu_status(unsafe { wwr_sys::wwr_ccl_group_end() });
    result.and(ended)
}

/// The root's half of a broadcast: `send` to every rank, on `stream`.
/// In place, `wwrcclBcast`, so the root only reads `send`; every other
/// rank calls [`broadcast_recv`] with this rank as the root and a buffer
/// of `send`'s length.
///
/// # Errors
/// The library's refusal, as a `GpuRuntime` error naming the status.
#[debug_requires(stream.dev_idx() == comm.dev_idx(), "the stream is on the rank's device")]
#[track_caller]
pub fn broadcast_send(comm: &Comm, send: &impl DeviceSlice<f64>, stream: &Stream) -> Status {
    // SAFETY: `send` is device memory of its `len()` elements until the
    // enqueued work finishes (DeviceSlice's promise); the root of an
    // in-place broadcast reads its buffer and writes none of it, so the
    // const pointer is never written through; the handle and the stream
    // are live.
    gpu_status(unsafe {
        wwr_sys::wwr_ccl_bcast(
            send.as_ptr().cast_mut().cast(),
            send.len(),
            wwr_ccl_data_type::WWR_CCL_FLOAT64,
            rank_i32(comm.rank),
            comm.raw,
            stream.stream(),
        )
    })
}

/// A rank's half of a broadcast from `root`: `recv` written with the
/// root's buffer, on `stream`.
///
/// # Errors
/// The library's refusal, as a `GpuRuntime` error naming the status.
#[requires(lt!(root, comm.ranks()), "the root is a rank of the clique")]
// RELEASE: no memory rests on it: a root that calls this half passes
// its own buffer to the library as any rank does, and only the result is
// wrong.
#[debug_requires(root != comm.rank(), "the root calls broadcast_send")]
#[debug_requires(stream.dev_idx() == comm.dev_idx(), "the stream is on the rank's device")]
#[track_caller]
pub fn broadcast_recv(
    comm: &Comm,
    recv: &mut impl DeviceSliceMut<f64>,
    root: usize,
    stream: &Stream,
) -> Status {
    let len = recv.len();
    let ptr = recv.as_mut_ptr();
    // SAFETY: `recv` is device memory of `len` elements with no other user
    // until the enqueued work finishes (DeviceSliceMut's promise), and the
    // library writes `len` of them; the handle and the stream are live.
    gpu_status(unsafe {
        wwr_sys::wwr_ccl_bcast(
            ptr.cast(),
            len,
            wwr_ccl_data_type::WWR_CCL_FLOAT64,
            rank_i32(root),
            comm.raw,
            stream.stream(),
        )
    })
}

/// A rank's half of a sum to `root`, not the root itself: `send` added
/// into the root's buffer, on `stream`. The root calls [`reduce_recv`].
///
/// The sum's order is the library's, fixed by its algorithm and the
/// clique's size, so it differs from one rank's own sum by rounding: an
/// unordered reduce (`docs/programming_model.md`, "An edge declares").
///
/// # Errors
/// The library's refusal, as a `GpuRuntime` error naming the status.
#[requires(lt!(root, comm.ranks()), "the root is a rank of the clique")]
// RELEASE: no memory rests on it: a root that calls this half passes
// its own buffer to the library as any rank does, and only the result is
// wrong.
#[debug_requires(root != comm.rank(), "the root calls reduce_recv")]
#[debug_requires(stream.dev_idx() == comm.dev_idx(), "the stream is on the rank's device")]
#[track_caller]
pub fn reduce_send(
    comm: &Comm,
    send: &impl DeviceSlice<f64>,
    root: usize,
    stream: &Stream,
) -> Status {
    // SAFETY: `send` is device memory of its `len()` elements until the
    // enqueued work finishes (DeviceSlice's promise); a rank that is not
    // the root writes no receive buffer, so none is passed; the handle and
    // the stream are live.
    gpu_status(unsafe {
        wwr_sys::wwr_ccl_reduce(
            send.as_ptr().cast(),
            std::ptr::null_mut(),
            send.len(),
            wwr_ccl_data_type::WWR_CCL_FLOAT64,
            wwr_ccl_red_op::WWR_CCL_SUM,
            rank_i32(root),
            comm.raw,
            stream.stream(),
        )
    })
}

/// The root's half of a sum: every rank's `send`, this one's included,
/// summed into `recv`, on `stream`. Unordered, as [`reduce_send`] says.
///
/// # Errors
/// The library's refusal, as a `GpuRuntime` error naming the status.
#[requires(eq!(recv.len(), send.len()), "the two buffers are one length")]
#[debug_requires(stream.dev_idx() == comm.dev_idx(), "the stream is on the rank's device")]
#[track_caller]
pub fn reduce_recv(
    comm: &Comm,
    send: &impl DeviceSlice<f64>,
    recv: &mut impl DeviceSliceMut<f64>,
    stream: &Stream,
) -> Status {
    let ptr = recv.as_mut_ptr();
    // SAFETY: both are device memory of `send.len()` elements, checked one
    // length above, until the enqueued work finishes, `recv` with no other
    // user, so apart from `send` (DeviceSlice's and DeviceSliceMut's
    // promises); the handle and the stream are live.
    gpu_status(unsafe {
        wwr_sys::wwr_ccl_reduce(
            send.as_ptr().cast(),
            ptr.cast(),
            send.len(),
            wwr_ccl_data_type::WWR_CCL_FLOAT64,
            wwr_ccl_red_op::WWR_CCL_SUM,
            rank_i32(comm.rank),
            comm.raw,
            stream.stream(),
        )
    })
}

/// Every rank's `send` summed into every rank's `recv`, on `stream`.
/// Unordered, as [`reduce_send`] says, and the same on every rank.
///
/// # Errors
/// The library's refusal, as a `GpuRuntime` error naming the status.
#[requires(eq!(recv.len(), send.len()), "the two buffers are one length")]
#[debug_requires(stream.dev_idx() == comm.dev_idx(), "the stream is on the rank's device")]
#[track_caller]
pub fn all_reduce(
    comm: &Comm,
    send: &impl DeviceSlice<f64>,
    recv: &mut impl DeviceSliceMut<f64>,
    stream: &Stream,
) -> Status {
    let ptr = recv.as_mut_ptr();
    // SAFETY: as in `reduce_recv`.
    gpu_status(unsafe {
        wwr_sys::wwr_ccl_all_reduce(
            send.as_ptr().cast(),
            ptr.cast(),
            send.len(),
            wwr_ccl_data_type::WWR_CCL_FLOAT64,
            wwr_ccl_red_op::WWR_CCL_SUM,
            comm.raw,
            stream.stream(),
        )
    })
}

/// Every rank's `send` written into every rank's `recv`, rank `r`'s at
/// `r` times `send`'s length, on `stream`: the gather of equal pieces,
/// exact. Every rank's `send` is one length.
///
/// # Errors
/// The library's refusal, as a `GpuRuntime` error naming the status.
#[requires(
    eq!(Some(recv.len()), send.len().checked_mul(comm.ranks())),
    "recv holds a piece per rank"
)]
#[debug_requires(stream.dev_idx() == comm.dev_idx(), "the stream is on the rank's device")]
#[track_caller]
pub fn all_gather(
    comm: &Comm,
    send: &impl DeviceSlice<f64>,
    recv: &mut impl DeviceSliceMut<f64>,
    stream: &Stream,
) -> Status {
    let ptr = recv.as_mut_ptr();
    // SAFETY: `send` is device memory of its `len()` elements and `recv`
    // of `ranks` times that, checked above, until the enqueued work
    // finishes, `recv` with no other user, so apart from `send`
    // (DeviceSlice's and DeviceSliceMut's promises); the handle and the
    // stream are live.
    gpu_status(unsafe {
        wwr_sys::wwr_ccl_all_gather(
            send.as_ptr().cast(),
            ptr.cast(),
            send.len(),
            wwr_ccl_data_type::WWR_CCL_FLOAT64,
            comm.raw,
            stream.stream(),
        )
    })
}

/// `send` sent to rank `peer`, on `stream`; `peer` calls [`recv`] from
/// this rank with a buffer of `send`'s length. Two calls between a pair
/// match in the order each issues them; a rank exchanging with several
/// peers issues its calls in one [`group`].
///
/// # Errors
/// The library's refusal, as a `GpuRuntime` error naming the status.
#[requires(lt!(peer, comm.ranks()), "the peer is a rank of the clique")]
#[debug_requires(stream.dev_idx() == comm.dev_idx(), "the stream is on the rank's device")]
#[track_caller]
pub fn send(comm: &Comm, send: &impl DeviceSlice<f64>, peer: usize, stream: &Stream) -> Status {
    // SAFETY: `send` is device memory of its `len()` elements until the
    // enqueued work finishes (DeviceSlice's promise); the handle and the
    // stream are live.
    gpu_status(unsafe {
        wwr_sys::wwr_ccl_send(
            send.as_ptr().cast(),
            send.len(),
            wwr_ccl_data_type::WWR_CCL_FLOAT64,
            rank_i32(peer),
            comm.raw,
            stream.stream(),
        )
    })
}

/// `recv` written with what rank `peer` [`send`]s this rank, on `stream`.
///
/// # Errors
/// The library's refusal, as a `GpuRuntime` error naming the status.
#[requires(lt!(peer, comm.ranks()), "the peer is a rank of the clique")]
#[debug_requires(stream.dev_idx() == comm.dev_idx(), "the stream is on the rank's device")]
#[track_caller]
pub fn recv(
    comm: &Comm,
    recv: &mut impl DeviceSliceMut<f64>,
    peer: usize,
    stream: &Stream,
) -> Status {
    let len = recv.len();
    let ptr = recv.as_mut_ptr();
    // SAFETY: `recv` is device memory of `len` elements with no other user
    // until the enqueued work finishes (DeviceSliceMut's promise), and the
    // library writes `len` of them; the handle and the stream are live.
    gpu_status(unsafe {
        wwr_sys::wwr_ccl_recv(
            ptr.cast(),
            len,
            wwr_ccl_data_type::WWR_CCL_FLOAT64,
            rank_i32(peer),
            comm.raw,
            stream.stream(),
        )
    })
}

/// A rank as the C ABI takes it. Every rank is below the clique's size,
/// which [`Comm::init_all`] held to an `i32`.
fn rank_i32(rank: usize) -> i32 {
    i32::try_from(rank).unwrap_or(i32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::ScopedDeviceIndex;
    use crate::runtime::device_count;
    use crate::runtime::tests::{DeviceBytes, ok};

    /// One rank's device, its stream and its buffers, each made with the
    /// rank's device current.
    struct Rank {
        dev: i32,
        stream: Stream,
    }

    impl Rank {
        fn new(dev: i32) -> Self {
            Rank {
                dev,
                stream: Stream::new(dev),
            }
        }

        fn upload(&self, host: &[f64]) -> DeviceBytes {
            let _current = ScopedDeviceIndex::new(self.dev);
            DeviceBytes::upload(host, &self.stream)
        }

        fn zeros(&self, n: usize) -> DeviceBytes {
            self.upload(&vec![0.0; n])
        }

        /// The buffer's doubles, after the work on the stream.
        #[allow(
            clippy::disallowed_methods,
            reason = "a test reads its result back on the host: the harness waits, not a task"
        )]
        fn read(&self, buf: &DeviceBytes) -> Vec<f64> {
            let _current = ScopedDeviceIndex::new(self.dev);
            ok(self.stream.sync());
            buf.read(&self.stream)
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_ne_bytes(*b))
                .collect()
        }
    }

    /// Rank `r`'s piece of the tests' data: `n` integers, so every sum of
    /// them is exact in any order and the comparisons are bitwise.
    fn piece(r: usize, n: usize) -> Vec<f64> {
        (0..n).map(|i| (100 * r + i) as f64).collect()
    }

    /// Every collective on `n` ranks, devices `0..n`, issued from this one
    /// thread in a group per collective, each compared bitwise with the
    /// host's answer: integers, so a sum in any order is exact.
    fn every_collective(n: usize) {
        let devices: Vec<i32> = (0..n).map(|d| d as i32).collect();
        let comms = Comm::init_all(&devices).unwrap();
        let ranks: Vec<Rank> = devices.iter().map(|&d| Rank::new(d)).collect();
        assert!(
            comms
                .iter()
                .enumerate()
                .all(|(r, c)| c.rank() == r && c.ranks() == n)
        );
        let len = 5;

        // Broadcast from the last rank.
        let root = n - 1;
        let mut bufs: Vec<DeviceBytes> = (0..n)
            .map(|r| {
                ranks[r].upload(&if r == root {
                    piece(r, len)
                } else {
                    vec![-1.0; len]
                })
            })
            .collect();
        group(|| {
            for (r, buf) in bufs.iter_mut().enumerate() {
                if r == root {
                    broadcast_send(&comms[r], buf, &ranks[r].stream)?;
                } else {
                    broadcast_recv(&comms[r], buf, root, &ranks[r].stream)?;
                }
            }
            Ok(())
        })
        .unwrap();
        for r in 0..n {
            assert_eq!(
                ranks[r].read(&bufs[r]),
                piece(root, len),
                "broadcast, rank {r}"
            );
        }

        // All-gather, reduce to rank 0 and all-reduce of every rank's piece.
        let sends: Vec<DeviceBytes> = (0..n).map(|r| ranks[r].upload(&piece(r, len))).collect();
        let mut gathered: Vec<DeviceBytes> = ranks.iter().map(|k| k.zeros(n * len)).collect();
        let mut summed: Vec<DeviceBytes> = ranks.iter().map(|k| k.zeros(len)).collect();
        let mut root_sum = ranks[0].zeros(len);
        group(|| {
            for r in 0..n {
                all_gather(&comms[r], &sends[r], &mut gathered[r], &ranks[r].stream)?;
            }
            Ok(())
        })
        .unwrap();
        group(|| {
            for r in 0..n {
                if r == 0 {
                    reduce_recv(&comms[0], &sends[0], &mut root_sum, &ranks[0].stream)?;
                } else {
                    reduce_send(&comms[r], &sends[r], 0, &ranks[r].stream)?;
                }
            }
            Ok(())
        })
        .unwrap();
        group(|| {
            for r in 0..n {
                all_reduce(&comms[r], &sends[r], &mut summed[r], &ranks[r].stream)?;
            }
            Ok(())
        })
        .unwrap();
        let all: Vec<f64> = (0..n).flat_map(|r| piece(r, len)).collect();
        let sum: Vec<f64> = (0..len)
            .map(|i| (0..n).map(|r| piece(r, len)[i]).sum())
            .collect();
        assert_eq!(ranks[0].read(&root_sum), sum, "reduce");
        for r in 0..n {
            assert_eq!(ranks[r].read(&gathered[r]), all, "all_gather, rank {r}");
            assert_eq!(ranks[r].read(&summed[r]), sum, "all_reduce, rank {r}");
        }

        // Every rank but 0 sends its piece, of its own length, to rank 0,
        // which receives each into a buffer of its own: a gather of
        // uneven pieces.
        let lens: Vec<usize> = (0..n).map(|r| r + 1).collect();
        let pieces: Vec<DeviceBytes> = (0..n)
            .map(|r| ranks[r].upload(&piece(r, lens[r])))
            .collect();
        let mut received: Vec<DeviceBytes> = lens.iter().map(|&l| ranks[0].zeros(l)).collect();
        group(|| {
            for r in 1..n {
                send(&comms[r], &pieces[r], 0, &ranks[r].stream)?;
                recv(&comms[0], &mut received[r], r, &ranks[0].stream)?;
            }
            Ok(())
        })
        .unwrap();
        for r in 1..n {
            assert_eq!(
                ranks[0].read(&received[r]),
                piece(r, lens[r]),
                "send, rank {r}"
            );
        }
    }

    #[test]
    fn a_status_has_a_name_and_a_description() {
        let error = gpu_status(wwr_ccl_result::WWR_CCL_RESULT_INVALID_USAGE).unwrap_err();
        assert_eq!(
            error.message,
            "WWR_CCL_RESULT_INVALID_USAGE (the caller used the library wrongly)"
        );
        assert_eq!(gpu_status(wwr_ccl_result::WWR_CCL_RESULT_SUCCESS), Ok(()));
    }

    /// Needs a GPU. A clique of one rank: every collective is a copy.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn every_collective_on_one_rank_is_a_copy() {
        every_collective(1);
    }

    /// Needs two GPUs, and runs on every device visible: every collective
    /// across them, bitwise against the host.
    #[test]
    #[ignore = "tier 5: needs two GPUs"]
    fn tier5_every_collective_across_the_visible_devices() {
        let n = device_count().unwrap();
        assert!(n >= 2, "{n} devices");
        println!("ccl: {n} ranks");
        every_collective(n);
    }

    /// Needs two GPUs. A device listed twice is the library's refusal, a
    /// value, not an abort.
    #[test]
    #[ignore = "tier 5: needs two GPUs"]
    fn tier5_a_device_listed_twice_is_refused() {
        let error = Comm::init_all(&[0, 0]).err().expect("a refusal");
        assert!(
            error.message.starts_with("WWR_CCL_RESULT_"),
            "{}",
            error.message
        );
    }
}
