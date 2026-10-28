//! Host-side FP16 packing. The scalar arithmetic defines the bit contract.

/// f32 -> f16 bits (round to nearest even), no external crates.
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let mant = b & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if mant != 0 { 1 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let mut hm = half as u16;
        if rem > halfway || (rem == halfway && (half & 1) == 1) {
            hm += 1;
        }
        return sign | hm;
    }
    let mut hm = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (hm & 1) == 1) {
        hm += 1;
    }
    sign | hm as u16
}

pub fn pack_f16x2(values: &[f32]) -> Vec<u32> {
    pack_parts(&[values])
}

pub fn pack_parts(parts: &[&[f32]]) -> Vec<u32> {
    for part in parts {
        assert_eq!(part.len() % 2, 0, "FP16 pairs require even-length parts");
    }
    let mut output = Vec::<u32>::with_capacity(parts.iter().map(|p| p.len() / 2).sum());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let vectorized = std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("f16c");
    for part in parts {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if vectorized {
            let begin = output.len();
            let count = part.len() / 2;
            // SAFETY: feature detection also checks OS AVX state support. The
            // spare slice has count words; pack_f16c initializes every word.
            unsafe {
                pack_f16c(part, &mut output.spare_capacity_mut()[..count]);
                output.set_len(begin + count);
            }
            continue;
        }
        output.extend(
            part.chunks_exact(2).map(|c| {
                u32::from(f32_to_f16_bits(c[0])) | (u32::from(f32_to_f16_bits(c[1])) << 16)
            }),
        );
    }
    output
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,f16c")]
unsafe fn pack_f16c(values: &[f32], output: &mut [std::mem::MaybeUninit<u32>]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let bulk = values.len() / 8 * 8;
    for i in (0..bulk).step_by(8) {
        // SAFETY: unaligned loads/stores are intentional. Eight f32 inputs
        // become four initialized packed words inside the supplied slices.
        unsafe {
            let floats = _mm256_loadu_ps(values.as_ptr().add(i));
            let halves = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(floats);
            _mm_storeu_si128(output.as_mut_ptr().add(i / 2).cast(), halves);
            // F16C preserves NaN payloads and quiets signaling NaNs, whereas
            // the original scalar contract uses sign | 0x7c01. Repair only
            // blocks containing NaNs, without penalizing finite model weights.
            if _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_UNORD_Q>(floats, floats)) != 0 {
                for j in 0..4 {
                    let a = values[i + j * 2];
                    let b = values[i + j * 2 + 1];
                    if a.is_nan() || b.is_nan() {
                        output[i / 2 + j].write(
                            u32::from(f32_to_f16_bits(a)) | (u32::from(f32_to_f16_bits(b)) << 16),
                        );
                    }
                }
            }
        }
    }
    for (j, c) in values[bulk..].chunks_exact(2).enumerate() {
        output[bulk / 2 + j]
            .write(u32::from(f32_to_f16_bits(c[0])) | (u32::from(f32_to_f16_bits(c[1])) << 16));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(values: &[f32]) {
        let expected: Vec<u32> = values
            .chunks_exact(2)
            .map(|c| u32::from(f32_to_f16_bits(c[0])) | (u32::from(f32_to_f16_bits(c[1])) << 16))
            .collect();
        assert_eq!(pack_f16x2(values), expected);
    }

    #[test]
    fn special_bits_and_rounding_boundaries() {
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16_bits(f32::from_bits(0x7fc01234)), 0x7c01);
        assert_eq!(f32_to_f16_bits(f32::from_bits(0xffc01234)), 0xfc01);
        let mut values = Vec::new();
        // Cover every normal half value and its fp32 half-way neighbors.
        for exp in 1..31u32 {
            for mant in 0..1024u32 {
                let bits = ((exp + 112) << 23) | (mant << 13);
                for delta in [0u32, 0xfff, 0x1000, 0x1001, 0x1fff] {
                    values.extend([
                        f32::from_bits(bits + delta),
                        f32::from_bits((bits + delta) | 0x80000000),
                    ]);
                }
            }
        }
        for exp in 0..114u32 {
            for mant in [0, 1, 0x3fffff, 0x7fffff] {
                let bits = (exp << 23) | mant;
                values.extend([f32::from_bits(bits), f32::from_bits(bits | 0x80000000)]);
            }
        }
        for h in 0..1024u32 {
            let value = h as f32 * 2.0f32.powi(-24);
            assert_eq!(f32_to_f16_bits(value), h as u16);
            for v in [value, value + 2.0f32.powi(-25), value - 2.0f32.powi(-25)] {
                values.extend([v, -v]);
            }
        }
        for bits in [
            0u32, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc01234, 0x7f800001,
            0x7fffffff,
        ] {
            values.extend([f32::from_bits(bits); 8]);
        }
        check(&values);
    }

    #[test]
    fn random_bits_parts_and_tails() {
        let mut state = 0x12ab34cdu32;
        let values: Vec<f32> = (0..200_000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                f32::from_bits(state)
            })
            .collect();
        check(&values);
        for len in [0usize, 2, 4, 6, 8, 10, 14, 16, 18, 30, 64, 1026] {
            check(&values[1..len + 1]);
            let parts = [&values[1..len + 1], &values[3..9], &[]];
            let joined = parts.concat();
            assert_eq!(pack_parts(&parts), pack_f16x2(&joined));
        }
    }

    #[test]
    #[should_panic(expected = "even-length")]
    fn odd_part_is_rejected() {
        pack_parts(&[&[1.0], &[2.0]]);
    }
}
