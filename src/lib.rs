extern crate convolution;
use std::mem;

use convolution::convolution;

fn construct_array(input: *mut f64, size: usize) -> Vec<f64> {
    unsafe {
        Vec::from_raw_parts(input, size, size)
    }
}

#[no_mangle]
pub extern fn convolution_cffi(input_a: *mut f64, a_size: usize,
                    input_b: *mut f64, b_size: usize,
                    o_size: usize) -> *const f64 {
    // Convert to vectors
    let a = construct_array(input_a, a_size);
    let b = construct_array(input_b, b_size);

    let out = convolution(&a, &b);
    assert_eq!(out.len() as usize, o_size);

    let out_ptr = out.as_ptr();
    mem::forget(out);

    out_ptr
}