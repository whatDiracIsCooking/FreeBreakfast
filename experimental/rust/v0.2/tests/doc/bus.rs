# use fb_core_error_handling::{Result, Status};
# use fb_v02::budget::{Backing, Budget, Caps, Ledger};
# use fb_v02::flow::{Bus, Next, Placement, Slot, Step, View};
# struct TestBus { ledger: Ledger, source: Slot<f64, ()>, reader: Slot<f64, ()> }
# impl Bus for TestBus {
#     type Shared = ();
#     type Lane = ();
#     type Guard = ();
#     fn shared(&self) -> &() { &() }
#     fn ledger(&self) -> &Ledger { &self.ledger }
#     fn enter(&self, _: Placement) {}
#     fn make_lane(&self, _: Placement) -> Result<()> { Ok(()) }
#     fn leave(&self, _: &()) -> Status { Ok(()) }
#     fn bind_workspace(&self, _: usize, _: u64) -> Status { Ok(()) }
# }
# #[derive(Clone, Copy)]
# struct Source;
# impl Step<TestBus> for Source {
#     const NAME: &'static str = "source";
#     type Output = f64;
#     type Reads = ();
#     fn slot(bus: &TestBus) -> &Slot<f64, ()> { &bus.source }
#     fn mem_budget(_: &(), _: &(), _: &Budget<'_>) -> Result<f64> { Ok(1.0) }
#     fn run(_: &mut f64, _: &View<'_, TestBus, Self>) -> Result<Next> { Ok(Next::Done) }
# }
