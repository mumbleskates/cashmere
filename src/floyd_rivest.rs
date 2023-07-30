use std::cmp::Ordering::{self, Greater, Less};
use std::{cmp, ptr};

pub fn select<T, F>(array: &mut [T], k: usize, mut f: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    let r = array.len() - 1;
    select_(array, &mut f, 0, r, k)
}

const A: usize = 600;
const B: f32 = 0.5;

fn select_<T, F>(array: &mut [T], cmp: &mut F, mut left: usize, mut right: usize, k: usize)
where
    F: FnMut(&T, &T) -> Ordering,
{
    let array = array;
    while right > left {
        if right - left > A {
            let n = (right - left + 1) as f32;
            let i = (k - left + 1) as f32;
            let z = n.ln();
            let s = B * (z * (2.0 / 3.0)).exp();
            let sn = s / n;
            let sd = B * (z * s * (1.0 - sn)).sqrt() * (i - n * 0.5).signum();

            let isn = i * s / n;
            let inner = k as f32 - isn + sd;
            let new_left = cmp::max(left, inner as usize);
            let new_right = cmp::min(right, (inner + s) as usize);

            select_(array, cmp, new_left, new_right, k)
        }

        let mut i = left + 1;
        let mut j = right - 1;
        array.swap(left, k);
        let t_idx = if cmp(&array[left], &array[right]) != Less {
            array.swap(left, right);
            right
        } else {
            left
        };

        // Need to do this without borrowing (but the assertion above ensures this doesn't alias)
        let arr_ptr = array.as_mut_ptr();
        let t: *const T = unsafe { arr_ptr.add(t_idx) };
        unsafe {
            // This code has been modified throughout to use pointer addition rather than
            // `array.get_unchecked_mut(x)` as the latter causes a dereference of `t` to alias with
            // the mutable borrow of `&mut array` that requires.
            while cmp(&*arr_ptr.add(i), &*t) == Less {
                i += 1
            }
            while cmp(&*arr_ptr.add(j), &*t) == Greater {
                j -= 1
            }
        }

        if i < j {
            // i < j, and i and j move toward each other, so this
            // assertion ensures that all indexing here is in-bounds.
            assert!(j < array.len());

            // FIXME: this unsafe code *should* be unnecessary: the
            // assertions above mean that LLVM could theoretically
            // optimise out the bounds checks, but it doesn't seem to
            // at the moment (it still does not, 2023-07-29).
            unsafe {
                while i < j {
                    ptr::swap(arr_ptr.add(i), arr_ptr.add(j));
                    i += 1;
                    j -= 1;
                    while cmp(&*arr_ptr.add(i), &*t) == Less {
                        i += 1
                    }
                    while cmp(&*arr_ptr.add(j), &*t) == Greater {
                        j -= 1
                    }
                }
            }
        }

        if left == t_idx {
            array.swap(left, j);
        } else {
            j += 1;
            array.swap(right, j);
        }
        if j <= k {
            left = j + 1
        }
        if k <= j {
            right = j.saturating_sub(1);
        }
    }
}
