//! Deterministic random sampling utilities for the Gumbel search backend.

#[derive(Clone, Debug)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    #[must_use]
    pub(crate) const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[must_use]
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    #[must_use]
    pub(crate) fn next_open01(&mut self) -> f32 {
        let bits = self.next_u64() >> 40;
        ((bits as f64 + 0.5) / ((1_u64 << 24) as f64)) as f32
    }

    #[must_use]
    pub(crate) fn next_gumbel(&mut self) -> f32 {
        let uniform = self.next_open01();
        -(-uniform.ln()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::SplitMix64;

    #[test]
    fn splitmix64_is_deterministic_for_same_seed() {
        let mut left = SplitMix64::new(7);
        let mut right = SplitMix64::new(7);

        for _ in 0..16 {
            assert_eq!(left.next_u64(), right.next_u64());
        }
    }

    #[test]
    fn gumbel_samples_are_finite() {
        let mut rng = SplitMix64::new(11);

        for _ in 0..32 {
            assert!(rng.next_gumbel().is_finite());
        }
    }
}
