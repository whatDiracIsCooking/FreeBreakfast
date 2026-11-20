// Generated: `Has` and `ReadSet` for tuples of one to eight steps (flow.rs includes this).
impl<T0> Has<T0, At<0>> for (T0,) {}
impl<B: Bus, T0: Step<B>> ReadSet<B> for (T0,) {
    fn names() -> Vec<&'static str> {
        vec![T0::NAME]
    }
}
impl<T0, T1> Has<T0, At<0>> for (T0, T1,) {}
impl<T0, T1> Has<T1, At<1>> for (T0, T1,) {}
impl<B: Bus, T0: Step<B>, T1: Step<B>> ReadSet<B> for (T0, T1,) {
    fn names() -> Vec<&'static str> {
        vec![T0::NAME, T1::NAME]
    }
}
impl<T0, T1, T2> Has<T0, At<0>> for (T0, T1, T2,) {}
impl<T0, T1, T2> Has<T1, At<1>> for (T0, T1, T2,) {}
impl<T0, T1, T2> Has<T2, At<2>> for (T0, T1, T2,) {}
impl<B: Bus, T0: Step<B>, T1: Step<B>, T2: Step<B>> ReadSet<B> for (T0, T1, T2,) {
    fn names() -> Vec<&'static str> {
        vec![T0::NAME, T1::NAME, T2::NAME]
    }
}
impl<T0, T1, T2, T3> Has<T0, At<0>> for (T0, T1, T2, T3,) {}
impl<T0, T1, T2, T3> Has<T1, At<1>> for (T0, T1, T2, T3,) {}
impl<T0, T1, T2, T3> Has<T2, At<2>> for (T0, T1, T2, T3,) {}
impl<T0, T1, T2, T3> Has<T3, At<3>> for (T0, T1, T2, T3,) {}
impl<B: Bus, T0: Step<B>, T1: Step<B>, T2: Step<B>, T3: Step<B>> ReadSet<B> for (T0, T1, T2, T3,) {
    fn names() -> Vec<&'static str> {
        vec![T0::NAME, T1::NAME, T2::NAME, T3::NAME]
    }
}
impl<T0, T1, T2, T3, T4> Has<T0, At<0>> for (T0, T1, T2, T3, T4,) {}
impl<T0, T1, T2, T3, T4> Has<T1, At<1>> for (T0, T1, T2, T3, T4,) {}
impl<T0, T1, T2, T3, T4> Has<T2, At<2>> for (T0, T1, T2, T3, T4,) {}
impl<T0, T1, T2, T3, T4> Has<T3, At<3>> for (T0, T1, T2, T3, T4,) {}
impl<T0, T1, T2, T3, T4> Has<T4, At<4>> for (T0, T1, T2, T3, T4,) {}
impl<B: Bus, T0: Step<B>, T1: Step<B>, T2: Step<B>, T3: Step<B>, T4: Step<B>> ReadSet<B> for (T0, T1, T2, T3, T4,) {
    fn names() -> Vec<&'static str> {
        vec![T0::NAME, T1::NAME, T2::NAME, T3::NAME, T4::NAME]
    }
}
impl<T0, T1, T2, T3, T4, T5> Has<T0, At<0>> for (T0, T1, T2, T3, T4, T5,) {}
impl<T0, T1, T2, T3, T4, T5> Has<T1, At<1>> for (T0, T1, T2, T3, T4, T5,) {}
impl<T0, T1, T2, T3, T4, T5> Has<T2, At<2>> for (T0, T1, T2, T3, T4, T5,) {}
impl<T0, T1, T2, T3, T4, T5> Has<T3, At<3>> for (T0, T1, T2, T3, T4, T5,) {}
impl<T0, T1, T2, T3, T4, T5> Has<T4, At<4>> for (T0, T1, T2, T3, T4, T5,) {}
impl<T0, T1, T2, T3, T4, T5> Has<T5, At<5>> for (T0, T1, T2, T3, T4, T5,) {}
impl<B: Bus, T0: Step<B>, T1: Step<B>, T2: Step<B>, T3: Step<B>, T4: Step<B>, T5: Step<B>> ReadSet<B> for (T0, T1, T2, T3, T4, T5,) {
    fn names() -> Vec<&'static str> {
        vec![T0::NAME, T1::NAME, T2::NAME, T3::NAME, T4::NAME, T5::NAME]
    }
}
impl<T0, T1, T2, T3, T4, T5, T6> Has<T0, At<0>> for (T0, T1, T2, T3, T4, T5, T6,) {}
impl<T0, T1, T2, T3, T4, T5, T6> Has<T1, At<1>> for (T0, T1, T2, T3, T4, T5, T6,) {}
impl<T0, T1, T2, T3, T4, T5, T6> Has<T2, At<2>> for (T0, T1, T2, T3, T4, T5, T6,) {}
impl<T0, T1, T2, T3, T4, T5, T6> Has<T3, At<3>> for (T0, T1, T2, T3, T4, T5, T6,) {}
impl<T0, T1, T2, T3, T4, T5, T6> Has<T4, At<4>> for (T0, T1, T2, T3, T4, T5, T6,) {}
impl<T0, T1, T2, T3, T4, T5, T6> Has<T5, At<5>> for (T0, T1, T2, T3, T4, T5, T6,) {}
impl<T0, T1, T2, T3, T4, T5, T6> Has<T6, At<6>> for (T0, T1, T2, T3, T4, T5, T6,) {}
impl<B: Bus, T0: Step<B>, T1: Step<B>, T2: Step<B>, T3: Step<B>, T4: Step<B>, T5: Step<B>, T6: Step<B>> ReadSet<B> for (T0, T1, T2, T3, T4, T5, T6,) {
    fn names() -> Vec<&'static str> {
        vec![T0::NAME, T1::NAME, T2::NAME, T3::NAME, T4::NAME, T5::NAME, T6::NAME]
    }
}
impl<T0, T1, T2, T3, T4, T5, T6, T7> Has<T0, At<0>> for (T0, T1, T2, T3, T4, T5, T6, T7,) {}
impl<T0, T1, T2, T3, T4, T5, T6, T7> Has<T1, At<1>> for (T0, T1, T2, T3, T4, T5, T6, T7,) {}
impl<T0, T1, T2, T3, T4, T5, T6, T7> Has<T2, At<2>> for (T0, T1, T2, T3, T4, T5, T6, T7,) {}
impl<T0, T1, T2, T3, T4, T5, T6, T7> Has<T3, At<3>> for (T0, T1, T2, T3, T4, T5, T6, T7,) {}
impl<T0, T1, T2, T3, T4, T5, T6, T7> Has<T4, At<4>> for (T0, T1, T2, T3, T4, T5, T6, T7,) {}
impl<T0, T1, T2, T3, T4, T5, T6, T7> Has<T5, At<5>> for (T0, T1, T2, T3, T4, T5, T6, T7,) {}
impl<T0, T1, T2, T3, T4, T5, T6, T7> Has<T6, At<6>> for (T0, T1, T2, T3, T4, T5, T6, T7,) {}
impl<T0, T1, T2, T3, T4, T5, T6, T7> Has<T7, At<7>> for (T0, T1, T2, T3, T4, T5, T6, T7,) {}
impl<B: Bus, T0: Step<B>, T1: Step<B>, T2: Step<B>, T3: Step<B>, T4: Step<B>, T5: Step<B>, T6: Step<B>, T7: Step<B>> ReadSet<B> for (T0, T1, T2, T3, T4, T5, T6, T7,) {
    fn names() -> Vec<&'static str> {
        vec![T0::NAME, T1::NAME, T2::NAME, T3::NAME, T4::NAME, T5::NAME, T6::NAME, T7::NAME]
    }
}
