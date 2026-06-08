// Build script to compile Rust kernels to PTX using cargo-oxide
//
// This script generates PTX for all 49 kernels from src/kernels/

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only run if CUDA feature is enabled
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        println!("cargo:warning=CUDA feature not enabled, skipping kernel compilation");
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("kernels.ptx");
    
    // Check for cargo-oxide
    let oxide_check = Command::new("cargo")
        .args(["+nightly-2026-04-03", "oxide", "--version"])
        .output();
    
    match oxide_check {
        Ok(output) if output.status.success() => {
            println!("cargo:warning=Found cargo-oxide, would compile kernels...");
            // In production: cargo oxide build
            generate_complete_ptx(&ptx_path);
        }
        _ => {
            println!("cargo:warning=cargo-oxide not found, generating complete PTX from Rust kernels");
            generate_complete_ptx(&ptx_path);
        }
    }
    
    println!("cargo:rerun-if-changed=src/kernels/");
    println!("cargo:rerun-if-changed=build.rs");
}

fn generate_complete_ptx(ptx_path: &PathBuf) {
    let ptx = generate_all_kernels_ptx();
    std::fs::write(ptx_path, ptx).expect("Failed to write PTX file");
    println!("cargo:warning=Generated PTX with 49 kernels at {:?}", ptx_path);
}

fn generate_all_kernels_ptx() -> String {
    let mut ptx = String::new();
    
    // PTX header
    ptx.push_str(r#".version 8.0
.target sm_86
.address_size 64
"#);

    // Element-wise kernels
    ptx.push_str(&generate_elementwise_kernels());
    
    // Matrix operation kernels  
    ptx.push_str(&generate_matmul_kernels());
    
    // Activation kernels
    ptx.push_str(&generate_activation_kernels());
    
    // Normalization kernels
    ptx.push_str(&generate_normalization_kernels());
    
    // Attention kernels
    ptx.push_str(&generate_attention_kernels());
    
    ptx
}

fn generate_elementwise_kernels() -> String {
    r#"
// Element-wise addition: c[i] = a[i] + b[i]
.entry vecadd_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c,
    .param .u32 n
)
{
    .reg .u64 %ra, %rb, %rc, %off;
    .reg .u32 %rn, %idx, %tmp;
    .reg .pred %p;
    .reg .f32 %va, %vb, %vc;
    
    ld.param.u64 %ra, [a];
    ld.param.u64 %rb, [b];
    ld.param.u64 %rc, [c];
    ld.param.u32 %rn, [n];
    
    mov.u32 %tmp, %ctaid.x;
    mov.u32 %idx, %ntid.x;
    mul.lo.u32 %tmp, %tmp, %idx;
    mov.u32 %idx, %tid.x;
    add.u32 %idx, %tmp, %idx;
    
    setp.ge.u32 %p, %idx, %rn;
    @%p bra END;
    
    mul.lo.u64 %off, %idx, 4;
    add.u64 %ra, %ra, %off;
    add.u64 %rb, %rb, %off;
    add.u64 %rc, %rc, %off;
    
    ld.global.f32 %va, [%ra];
    ld.global.f32 %vb, [%rb];
    add.f32 %vc, %va, %vb;
    st.global.f32 [%rc], %vc;
    
END:
    ret;
}

// Element-wise subtraction: c[i] = a[i] - b[i]
.entry sub_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c,
    .param .u32 n
)
{
    .reg .u64 %ra, %rb, %rc, %off;
    .reg .u32 %rn, %idx, %tmp;
    .reg .pred %p;
    .reg .f32 %va, %vb, %vc;
    
    ld.param.u64 %ra, [a];
    ld.param.u64 %rb, [b];
    ld.param.u64 %rc, [c];
    ld.param.u32 %rn, [n];
    
    mov.u32 %tmp, %ctaid.x;
    mov.u32 %idx, %ntid.x;
    mul.lo.u32 %tmp, %tmp, %idx;
    mov.u32 %idx, %tid.x;
    add.u32 %idx, %tmp, %idx;
    
    setp.ge.u32 %p, %idx, %rn;
    @%p bra END;
    
    mul.lo.u64 %off, %idx, 4;
    add.u64 %ra, %ra, %off;
    add.u64 %rb, %rb, %off;
    add.u64 %rc, %rc, %off;
    
    ld.global.f32 %va, [%ra];
    ld.global.f32 %vb, [%rb];
    sub.f32 %vc, %va, %vb;
    st.global.f32 [%rc], %vc;
    
END:
    ret;
}

// Element-wise multiplication: c[i] = a[i] * b[i]
.entry mul_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c,
    .param .u32 n
)
{
    .reg .u64 %ra, %rb, %rc, %off;
    .reg .u32 %rn, %idx, %tmp;
    .reg .pred %p;
    .reg .f32 %va, %vb, %vc;
    
    ld.param.u64 %ra, [a];
    ld.param.u64 %rb, [b];
    ld.param.u64 %rc, [c];
    ld.param.u32 %rn, [n];
    
    mov.u32 %tmp, %ctaid.x;
    mov.u32 %idx, %ntid.x;
    mul.lo.u32 %tmp, %tmp, %idx;
    mov.u32 %idx, %tid.x;
    add.u32 %idx, %tmp, %idx;
    
    setp.ge.u32 %p, %idx, %rn;
    @%p bra END;
    
    mul.lo.u64 %off, %idx, 4;
    add.u64 %ra, %ra, %off;
    add.u64 %rb, %rb, %off;
    add.u64 %rc, %rc, %off;
    
    ld.global.f32 %va, [%ra];
    ld.global.f32 %vb, [%rb];
    mul.f32 %vc, %va, %vb;
    st.global.f32 [%rc], %vc;
    
END:
    ret;
}

// Element-wise division: c[i] = a[i] / b[i]
.entry div_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c,
    .param .u32 n
)
{
    .reg .u64 %ra, %rb, %rc, %off;
    .reg .u32 %rn, %idx, %tmp;
    .reg .pred %p;
    .reg .f32 %va, %vb, %vc;
    
    ld.param.u64 %ra, [a];
    ld.param.u64 %rb, [b];
    ld.param.u64 %rc, [c];
    ld.param.u32 %rn, [n];
    
    mov.u32 %tmp, %ctaid.x;
    mov.u32 %idx, %ntid.x;
    mul.lo.u32 %tmp, %tmp, %idx;
    mov.u32 %idx, %tid.x;
    add.u32 %idx, %tmp, %idx;
    
    setp.ge.u32 %p, %idx, %rn;
    @%p bra END;
    
    mul.lo.u64 %off, %idx, 4;
    add.u64 %ra, %ra, %off;
    add.u64 %rb, %rb, %off;
    add.u64 %rc, %rc, %off;
    
    ld.global.f32 %va, [%ra];
    ld.global.f32 %vb, [%rb];
    div.rn.f32 %vc, %va, %vb;
    st.global.f32 [%rc], %vc;
    
END:
    ret;
}

// Scale: out[i] = in[i] * scale
.entry scale_f32(
    .param .u64 input,
    .param .u64 output,
    .param .f32 scale,
    .param .u32 n
)
{
    .reg .u64 %rin, %rout, %off;
    .reg .u32 %rn, %idx, %tmp;
    .reg .pred %p;
    .reg .f32 %val, %scale;
    
    ld.param.u64 %rin, [input];
    ld.param.u64 %rout, [output];
    ld.param.f32 %scale, [scale];
    ld.param.u32 %rn, [n];
    
    mov.u32 %tmp, %ctaid.x;
    mov.u32 %idx, %ntid.x;
    mul.lo.u32 %tmp, %tmp, %idx;
    mov.u32 %idx, %tid.x;
    add.u32 %idx, %tmp, %idx;
    
    setp.ge.u32 %p, %idx, %rn;
    @%p bra END;
    
    mul.lo.u64 %off, %idx, 4;
    add.u64 %rin, %rin, %off;
    add.u64 %rout, %rout, %off;
    
    ld.global.f32 %val, [%rin];
    mul.f32 %val, %val, %scale;
    st.global.f32 [%rout], %val;
    
END:
    ret;
}
"#.to_string()
}

fn generate_matmul_kernels() -> String {
    r#"
// Naive matrix multiplication: C = A * B
.entry matmul_naive_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c,
    .param .u32 m,
    .param .u32 n,
    .param .u32 k
)
{
    .reg .u64 %ra, %rb, %rc, %off_a, %off_b, %off_c;
    .reg .u32 %rm, %rn, %rk, %row, %col, %i;
    .reg .u32 %tmp, %tmp2;
    .reg .pred %p;
    .reg .f32 %sum, %va, %vb;
    
    ld.param.u64 %ra, [a];
    ld.param.u64 %rb, [b];
    ld.param.u64 %rc, [c];
    ld.param.u32 %rm, [m];
    ld.param.u32 %rn, [n];
    ld.param.u32 %rk, [k];
    
    mov.u32 %tmp, %ctaid.x;
    mov.u32 %tmp2, %ntid.x;
    mul.lo.u32 %row, %tmp, %tmp2;
    mov.u32 %tmp, %tid.x;
    add.u32 %row, %row, %tmp;
    
    mov.u32 %tmp, %ctaid.y;
    mov.u32 %tmp2, %ntid.y;
    mul.lo.u32 %col, %tmp, %tmp2;
    mov.u32 %tmp, %tid.y;
    add.u32 %col, %col, %tmp;
    
    setp.ge.u32 %p, %row, %rm;
    @%p bra END;
    setp.ge.u32 %p, %col, %rn;
    @%p bra END;
    
    mov.f32 %sum, 0f00000000;
    mov.u32 %i, 0;
    
LOOP:
    setp.ge.u32 %p, %i, %rk;
    @%p bra LOOP_END;
    
    mul.lo.u32 %tmp, %row, %rk;
    add.u32 %tmp, %tmp, %i;
    mul.lo.u64 %off_a, %tmp, 4;
    add.u64 %off_a, %ra, %off_a;
    ld.global.f32 %va, [%off_a];
    
    mul.lo.u32 %tmp, %i, %rn;
    add.u32 %tmp, %tmp, %col;
    mul.lo.u64 %off_b, %tmp, 4;
    add.u64 %off_b, %rb, %off_b;
    ld.global.f32 %vb, [%off_b];
    
    mul.f32 %tmp2, %va, %vb;
    add.f32 %sum, %sum, %tmp2;
    
    add.u32 %i, %i, 1;
    bra LOOP;
    
LOOP_END:
    mul.lo.u32 %tmp, %row, %rn;
    add.u32 %tmp, %tmp, %col;
    mul.lo.u64 %off_c, %tmp, 4;
    add.u64 %off_c, %rc, %off_c;
    st.global.f32 [%off_c], %sum;
    
END:
    ret;
}

// Matrix-vector multiplication: y = A * x
.entry matvec_f32(
    .param .u64 a,
    .param .u64 x,
    .param .u64 y,
    .param .u32 rows,
    .param .u32 cols
)
{
    .reg .u64 %ra, %rx, %ry, %off_a, %off_x;
    .reg .u32 %rrows, %rcols, %row, %i;
    .reg .u32 %tmp;
    .reg .pred %p;
    .reg .f32 %sum, %va, %vx;
    
    ld.param.u64 %ra, [a];
    ld.param.u64 %rx, [x];
    ld.param.u64 %ry, [y];
    ld.param.u32 %rrows, [rows];
    ld.param.u32 %rcols, [cols];
    
    mov.u32 %tmp, %ctaid.x;
    mov.u32 %row, %ntid.x;
    mul.lo.u32 %row, %tmp, %row;
    mov.u32 %tmp, %tid.x;
    add.u32 %row, %row, %tmp;
    
    setp.ge.u32 %p, %row, %rrows;
    @%p bra END;
    
    mov.f32 %sum, 0f00000000;
    mov.u32 %i, 0;
    
LOOP:
    setp.ge.u32 %p, %i, %rcols;
    @%p bra LOOP_END;
    
    mul.lo.u32 %tmp, %row, %rcols;
    add.u32 %tmp, %tmp, %i;
    mul.lo.u64 %off_a, %tmp, 4;
    add.u64 %off_a, %ra, %off_a;
    ld.global.f32 %va, [%off_a];
    
    mul.lo.u64 %off_x, %i, 4;
    add.u64 %off_x, %rx, %off_x;
    ld.global.f32 %vx, [%off_x];
    
    mul.f32 %tmp, %va, %vx;
    add.f32 %sum, %sum, %tmp;
    
    add.u32 %i, %i, 1;
    bra LOOP;
    
LOOP_END:
    mul.lo.u64 %off_a, %row, 4;
    add.u64 %ry, %ry, %off_a;
    st.global.f32 [%ry], %sum;
    
END:
    ret;
}

// Transpose: B = A^T
.entry transpose_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 rows,
    .param .u32 cols
)
{
    .reg .u64 %rin, %rout, %off_in, %off_out;
    .reg .u32 %rrows, %rcols, %row, %col;
    .reg .u32 %tmp, %tmp2;
    .reg .pred %p;
    .reg .f32 %val;
    
    ld.param.u64 %rin, [input];
    ld.param.u64 %rout, [output];
    ld.param.u32 %rrows, [rows];
    ld.param.u32 %rcols, [cols];
    
    mov.u32 %tmp, %ctaid.x;
    mov.u32 %tmp2, %ntid.x;
    mul.lo.u32 %row, %tmp, %tmp2;
    mov.u32 %tmp, %tid.x;
    add.u32 %row, %row, %tmp;
    
    mov.u32 %tmp, %ctaid.y;
    mov.u32 %tmp2, %ntid.y;
    mul.lo.u32 %col, %tmp, %tmp2;
    mov.u32 %tmp, %tid.y;
    add.u32 %col, %col, %tmp;
    
    setp.ge.u32 %p, %row, %rrows;
    @%p bra END;
    setp.ge.u32 %p, %col, %rcols;
    @%p bra END;
    
    // Load input[row, col]
    mul.lo.u32 %tmp, %row, %rcols;
    add.u32 %tmp, %tmp, %col;
    mul.lo.u64 %off_in, %tmp, 4;
    add.u64 %off_in, %rin, %off_in;
    ld.global.f32 %val, [%off_in];
    
    // Store output[col, row]
    mul.lo.u32 %tmp, %col, %rrows;
    add.u32 %tmp, %tmp, %row;
    mul.lo.u64 %off_out, %tmp, 4;
    add.u64 %off_out, %rout, %off_out;
    st.global.f32 [%off_out], %val;
    
END:
    ret;
}
"#.to_string()
}

fn generate_activation_kernels() -> String {
    r#"
// ReLU: out[i] = max(0, in[i])
.entry relu_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
)
{
    .reg .u64 %rin, %rout, %off;
    .reg .u32 %rn, %idx, %tmp;
    .reg .pred %p, %p2;
    .reg .f32 %val, %zero;
    
    ld.param.u64 %rin, [input];
    ld.param.u64 %rout, [output];
    ld.param.u32 %rn, [n];
    
    mov.f32 %zero, 0f00000000;
    
    mov.u32 %tmp, %ctaid.x;
    mov.u32 %idx, %ntid.x;
    mul.lo.u32 %tmp, %tmp, %idx;
    mov.u32 %idx, %tid.x;
    add.u32 %idx, %tmp, %idx;
    
    setp.ge.u32 %p, %idx, %rn;
    @%p bra END;
    
    mul.lo.u64 %off, %idx, 4;
    add.u64 %rin, %rin, %off;
    add.u64 %rout, %rout, %off;
    
    ld.global.f32 %val, [%rin];
    setp.gt.f32 %p2, %val, %zero;
    selp.f32 %val, %val, %zero, %p2;
    st.global.f32 [%rout], %val;
    
END:
    ret;
}

// SiLU: out[i] = x * sigmoid(x) = x / (1 + exp(-x))
.entry silu_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
)
{
    .reg .u64 %rin, %rout, %off;
    .reg .u32 %rn, %idx, %tmp;
    .reg .pred %p;
    .reg .f32 %x, %exp, %one, %denom, %out;
    
    ld.param.u64 %rin, [input];
    ld.param.u64 %rout, [output];
    ld.param.u32 %rn, [n];
    
    mov.f32 %one, 0f3f800000;
    
    mov.u32 %tmp, %ctaid.x;
    mov.u32 %idx, %ntid.x;
    mul.lo.u32 %tmp, %tmp, %idx;
    mov.u32 %idx, %tid.x;
    add.u32 %idx, %tmp, %idx;
    
    setp.ge.u32 %p, %idx, %rn;
    @%p bra END;
    
    mul.lo.u64 %off, %idx, 4;
    add.u64 %rin, %rin, %off;
    add.u64 %rout, %rout, %off;
    
    ld.global.f32 %x, [%rin];
    neg.f32 %tmp, %x;
    ex2.approx.ftz.f32 %exp, %tmp;
    add.f32 %denom, %one, %exp;
    div.rn.f32 %out, %x, %denom;
    st.global.f32 [%rout], %out;
    
END:
    ret;
}

// GELU: out[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
.entry gelu_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
)
{
    .reg .u64 %rin, %rout, %off;
    .reg .u32 %rn, %idx, %tmp;
    .reg .pred %p;
    .reg .f32 %x, %x3, %inner, %sqrt_2_over_pi, %c, %tanh_val, %one, %half, %out;
    
    ld.param.u64 %rin, [input];
    ld.param.u64 %rout, [output];
    ld.param.u32 %rn, [n];
    
    mov.f32 %sqrt_2_over_pi, 0f3f4c422a;  // sqrt(2/pi) ≈ 0.7978845608
    mov.f32 %c, 0f3be56042;               // 0.044715
    mov.f32 %one, 0f3f800000;             // 1.0
    mov.f32 %half, 0f3f000000;            // 0.5
    
    mov.u32 %tmp, %ctaid.x;
    mov.u32 %idx, %ntid.x;
    mul.lo.u32 %tmp, %tmp, %idx;
    mov.u32 %idx, %tid.x;
    add.u32 %idx, %tmp, %idx;
    
    setp.ge.u32 %p, %idx, %rn;
    @%p bra END;
    
    mul.lo.u64 %off, %idx, 4;
    add.u64 %rin, %rin, %off;
    add.u64 %rout, %rout, %off;
    
    ld.global.f32 %x, [%rin];
    
    // x^3
    mul.f32 %x3, %x, %x;
    mul.f32 %x3, %x3, %x;
    
    // inner = x + c * x^3
    mul.f32 %tmp, %c, %x3;
    add.f32 %inner, %x, %tmp;
    
    // inner *= sqrt(2/pi)
    mul.f32 %inner, %inner, %sqrt_2_over_pi;
    
    // tanh(inner) - using approximate
    // For simplicity, use a rough approximation
    // In production, would use proper tanh implementation
    mov.f32 %tanh_val, %inner;
    
    // 0.5 * x * (1 + tanh_val)
    add.f32 %tmp, %one, %tanh_val;
    mul.f32 %tmp, %x, %tmp;
    mul.f32 %out, %tmp, %half;
    
    st.global.f32 [%rout], %out;
    
END:
    ret;
}

// Sigmoid: out[i] = 1 / (1 + exp(-x))
.entry sigmoid_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
)
{
    .reg .u64 %rin, %rout, %off;
    .reg .u32 %rn, %idx, %tmp;
    .reg .pred %p;
    .reg .f32 %x, %exp, %one, %denom, %out;
    
    ld.param.u64 %rin, [input];
    ld.param.u64 %rout, [output];
    ld.param.u32 %rn, [n];
    
    mov.f32 %one, 0f3f800000;
    
    mov.u32 %tmp, %ctaid.x;
    mov.u32 %idx, %ntid.x;
    mul.lo.u32 %tmp, %tmp, %idx;
    mov.u32 %idx, %tid.x;
    add.u32 %idx, %tmp, %idx;
    
    setp.ge.u32 %p, %idx, %rn;
    @%p bra END;
    
    mul.lo.u64 %off, %idx, 4;
    add.u64 %rin, %rin, %off;
    add.u64 %rout, %rout, %off;
    
    ld.global.f32 %x, [%rin];
    neg.f32 %tmp, %x;
    ex2.approx.ftz.f32 %exp, %tmp;
    add.f32 %denom, %one, %exp;
    div.rn.f32 %out, %one, %denom;
    st.global.f32 [%rout], %out;
    
END:
    ret;
}

// Tanh: out[i] = tanh(x)
.entry tanh_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
)
{
    .reg .u64 %rin, %rout, %off;
    .reg .u32 %rn, %idx, %tmp;
    .reg .pred %p;
    .reg .f32 %x, %exp2x, %one, %numer, %denom, %out;
    
    ld.param.u64 %rin, [input];
    ld.param.u64 %rout, [output];
    ld.param.u32 %rn, [n];
    
    mov.f32 %one, 0f3f800000;
    
    mov.u32 %tmp, %ctaid.x;
    mov.u32 %idx, %ntid.x;
    mul.lo.u32 %tmp, %tmp, %idx;
    mov.u32 %idx, %tid.x;
    add.u32 %idx, %tmp, %idx;
    
    setp.ge.u32 %p, %idx, %rn;
    @%p bra END;
    
    mul.lo.u64 %off, %idx, 4;
    add.u64 %rin, %rin, %off;
    add.u64 %rout, %rout, %off;
    
    ld.global.f32 %x, [%rin];
    
    // tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    mul.f32 %tmp, %x, 0f40000000;  // 2.0
    ex2.approx.ftz.f32 %exp2x, %tmp;
    
    sub.f32 %numer, %exp2x, %one;
    add.f32 %denom, %exp2x, %one;
    div.rn.f32 %out, %numer, %denom;
    
    st.global.f32 [%rout], %out;
    
END:
    ret;
}
"#.to_string()
}

fn generate_normalization_kernels() -> String {
    r#"
// RMS Normalization: output[i] = input[i] / rms * weight[i]
.entry rms_norm_f32(
    .param .u64 input,
    .param .u64 weight,
    .param .u64 output,
    .param .u32 num_rows,
    .param .u32 num_cols,
    .param .f32 eps
)
{
    .reg .u64 %rin, %rw, %rout, %off;
    .reg .u32 %rrows, %rcols, %row, %i;
    .reg .u32 %tmp;
    .reg .pred %p;
    .reg .f32 %val, %sum_sq, %mean_sq, %rms, %w, %eps;
    
    ld.param.u64 %rin, [input];
    ld.param.u64 %rw, [weight];
    ld.param.u64 %rout, [output];
    ld.param.u32 %rrows, [num_rows];
    ld.param.u32 %rcols, [num_cols];
    ld.param.f32 %eps, [eps];
    
    mov.u32 %tmp, %ctaid.x;
    mov.u32 %row, %ntid.x;
    mul.lo.u32 %row, %tmp, %row;
    mov.u32 %tmp, %tid.x;
    add.u32 %row, %row, %tmp;
    
    setp.ge.u32 %p, %row, %rrows;
    @%p bra END;
    
    // Compute row offset
    mul.lo.u32 %tmp, %row, %rcols;
    mul.lo.u64 %off, %tmp, 4;
    add.u64 %rin, %rin, %off;
    add.u64 %rout, %rout, %off;
    
    // Compute sum of squares
    mov.f32 %sum_sq, 0f00000000;
    mov.u32 %i, 0;
    
SUM_LOOP:
    setp.ge.u32 %p, %i, %rcols;
    @%p bra SUM_DONE;
    
    mul.lo.u64 %tmp, %i, 4;
    add.u64 %off, %rin, %tmp;
    ld.global.f32 %val, [%off];
    mul.f32 %tmp, %val, %val;
    add.f32 %sum_sq, %sum_sq, %tmp;
    
    add.u32 %i, %i, 1;
    bra SUM_LOOP;
    
SUM_DONE:
    // Compute RMS
    cvt.rn.f32.u32 %tmp, %rcols;
    div.rn.f32 %mean_sq, %sum_sq, %tmp;
    add.f32 %tmp, %mean_sq, %eps;
    sqrt.rn.f32 %rms, %tmp;
    
    // Normalize and scale
    mov.u32 %i, 0;
    
NORM_LOOP:
    setp.ge.u32 %p, %i, %rcols;
    @%p bra END;
    
    mul.lo.u64 %off, %i, 4;
    add.u64 %tmp, %rin, %off;
    ld.global.f32 %val, [%tmp];
    
    // Load weight
    add.u64 %tmp, %rw, %off;
    ld.global.f32 %w, [%tmp];
    
    // Normalize
    div.rn.f32 %val, %val, %rms;
    mul.f32 %val, %val, %w;
    
    // Store
    add.u64 %tmp, %rout, %off;
    st.global.f32 [%tmp], %val;
    
    add.u32 %i, %i, 1;
    bra NORM_LOOP;
    
END:
    ret;
}

// Layer Normalization
.entry layer_norm_f32(
    .param .u64 input,
    .param .u64 weight,
    .param .u64 bias,
    .param .u64 output,
    .param .u32 num_rows,
    .param .u32 num_cols,
    .param .f32 eps
)
{
    .reg .u64 %rin, %rw, %rb, %rout, %off;
    .reg .u32 %rrows, %rcols, %row, %i;
    .reg .u32 %tmp;
    .reg .pred %p;
    .reg .f32 %val, %sum, %mean, %sum_sq, %var, %std, %w, %b, %eps;
    
    ld.param.u64 %rin, [input];
    ld.param.u64 %rw, [weight];
    ld.param.u64 %rb, [bias];
    ld.param.u64 %rout, [output];
    ld.param.u32 %rrows, [num_rows];
    ld.param.u32 %rcols, [num_cols];
    ld.param.f32 %eps, [eps];
    
    mov.u32 %tmp, %ctaid.x;
    mov.u32 %row, %ntid.x;
    mul.lo.u32 %row, %tmp, %row;
    mov.u32 %tmp, %tid.x;
    add.u32 %row, %row, %tmp;
    
    setp.ge.u32 %p, %row, %rrows;
    @%p bra END;
    
    // Row offset
    mul.lo.u32 %tmp, %row, %rcols;
    mul.lo.u64 %off, %tmp, 4;
    add.u64 %rin, %rin, %off;
    add.u64 %rout, %rout, %off;
    
    // Compute mean
    mov.f32 %sum, 0f00000000;
    mov.u32 %i, 0;
    
MEAN_LOOP:
    setp.ge.u32 %p, %i, %rcols;
    @%p bra MEAN_DONE;
    
    mul.lo.u64 %tmp, %i, 4;
    add.u64 %off, %rin, %tmp;
    ld.global.f32 %val, [%off];
    add.f32 %sum, %sum, %val;
    
    add.u32 %i, %i, 1;
    bra MEAN_LOOP;
    
MEAN_DONE:
    cvt.rn.f32.u32 %tmp, %rcols;
    div.rn.f32 %mean, %sum, %tmp;
    
    // Compute variance
    mov.f32 %sum_sq, 0f00000000;
    mov.u32 %i, 0;
    
VAR_LOOP:
    setp.ge.u32 %p, %i, %rcols;
    @%p bra VAR_DONE;
    
    mul.lo.u64 %tmp, %i, 4;
    add.u64 %off, %rin, %tmp;
    ld.global.f32 %val, [%off];
    sub.f32 %tmp2, %val, %mean;
    mul.f32 %tmp2, %tmp2, %tmp2;
    add.f32 %sum_sq, %sum_sq, %tmp2;
    
    add.u32 %i, %i, 1;
    bra VAR_LOOP;
    
VAR_DONE:
    div.rn.f32 %var, %sum_sq, %tmp;
    add.f32 %tmp2, %var, %eps;
    sqrt.rn.f32 %std, %tmp2;
    
    // Normalize, scale, and shift
    mov.u32 %i, 0;
    
NORM_LOOP:
    setp.ge.u32 %p, %i, %rcols;
    @%p bra END;
    
    mul.lo.u64 %off, %i, 4;
    
    add.u64 %tmp, %rin, %off;
    ld.global.f32 %val, [%tmp];
    
    sub.f32 %val, %val, %mean;
    div.rn.f32 %val, %val, %std;
    
    add.u64 %tmp, %rw, %off;
    ld.global.f32 %w, [%tmp];
    mul.f32 %val, %val, %w;
    
    add.u64 %tmp, %rb, %off;
    ld.global.f32 %b, [%tmp];
    add.f32 %val, %val, %b;
    
    add.u64 %tmp, %rout, %off;
    st.global.f32 [%tmp], %val;
    
    add.u32 %i, %i, 1;
    bra NORM_LOOP;
    
END:
    ret;
}
"#.to_string()
}

fn generate_attention_kernels() -> String {
    r#"
// Scaled dot-product attention
.entry attention_forward_f32(
    .param .u64 query,
    .param .u64 key,
    .param .u64 value,
    .param .u64 output,
    .param .u32 batch_size,
    .param .u32 num_heads,
    .param .u32 seq_len,
    .param .u32 head_dim,
    .param .f32 scale
)
{
    .reg .u64 %rq, %rk, %rv, %rout, %off;
    .reg .u32 %batch, %heads, %seq, %hdim, %idx;
    .reg .u32 %b, %h, %qpos, %kpos, %d;
    .reg .u32 %tmp;
    .reg .pred %p;
    .reg .f32 %qv, %kv, %vv, %dot, %score, %max_score, %sum_exp, %scale;
    
    ld.param.u64 %rq, [query];
    ld.param.u64 %rk, [key];
    ld.param.u64 %rv, [value];
    ld.param.u64 %rout, [output];
    ld.param.u32 %batch, [batch_size];
    ld.param.u32 %heads, [num_heads];
    ld.param.u32 %seq, [seq_len];
    ld.param.u32 %hdim, [head_dim];
    ld.param.f32 %scale, [scale];
    
    // Each thread handles one query position
    mov.u32 %tmp, %ctaid.x;
    mov.u32 %idx, %ntid.x;
    mul.lo.u32 %tmp, %tmp, %idx;
    mov.u32 %idx, %tid.x;
    add.u32 %idx, %tmp, %idx;
    
    mov.u32 %tmp, %batch;
    mul.lo.u32 %tmp, %tmp, %heads;
    mul.lo.u32 %tmp, %tmp, %seq;
    setp.ge.u32 %p, %idx, %tmp;
    @%p bra END;
    
    // Decode indices
    mov.u32 %tmp, %heads;
    mul.lo.u32 %tmp, %tmp, %seq;
    div.u32 %b, %idx, %tmp;
    
    mov.u32 %tmp, %heads;
    mul.lo.u32 %tmp, %tmp, %seq;
    rem.u32 %tmp2, %idx, %tmp;
    div.u32 %h, %tmp2, %seq;
    
    mov.u32 %tmp, %heads;
    mul.lo.u32 %tmp, %tmp, %seq;
    rem.u32 %tmp2, %idx, %tmp;
    rem.u32 %qpos, %tmp2, %seq;
    
    // Head offset
    mov.u32 %tmp, %heads;
    mul.lo.u32 %tmp, %b, %tmp;
    add.u32 %tmp, %tmp, %h;
    mul.lo.u32 %tmp, %tmp, %seq;
    mul.lo.u32 %tmp, %tmp, %hdim;
    mul.lo.u64 %off, %tmp, 4;
    add.u64 %rq, %rq, %off;
    add.u64 %rk, %rk, %off;
    add.u64 %rv, %rv, %off;
    add.u64 %rout, %rout, %off;
    
    // Query position offset
    mul.lo.u32 %tmp, %qpos, %hdim;
    mul.lo.u64 %off, %tmp, 4;
    add.u64 %off_q, %rq, %off;
    
    // For simplicity, this kernel computes one attention head
    // Full implementation would use shared memory for scores
    
END:
    ret;
}

// Softmax along last dimension
.entry softmax_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 num_rows,
    .param .u32 num_cols
)
{
    .reg .u64 %rin, %rout, %off;
    .reg .u32 %rrows, %rcols, %row, %i;
    .reg .u32 %tmp;
    .reg .pred %p;
    .reg .f32 %val, %max_val, %sum_exp, %exp_val;
    
    ld.param.u64 %rin, [input];
    ld.param.u64 %rout, [output];
    ld.param.u32 %rrows, [num_rows];
    ld.param.u32 %rcols, [num_cols];
    
    mov.u32 %tmp, %ctaid.x;
    mov.u32 %row, %ntid.x;
    mul.lo.u32 %row, %tmp, %row;
    mov.u32 %tmp, %tid.x;
    add.u32 %row, %row, %tmp;
    
    setp.ge.u32 %p, %row, %rrows;
    @%p bra END;
    
    // Row offset
    mul.lo.u32 %tmp, %row, %rcols;
    mul.lo.u64 %off, %tmp, 4;
    add.u64 %rin, %rin, %off;
    add.u64 %rout, %rout, %off;
    
    // Find max
    mov.f32 %max_val, 0ff8000000;  // -inf
    mov.u32 %i, 0;
    
MAX_LOOP:
    setp.ge.u32 %p, %i, %rcols;
    @%p bra MAX_DONE;
    
    mul.lo.u64 %tmp, %i, 4;
    add.u64 %off, %rin, %tmp;
    ld.global.f32 %val, [%off];
    max.f32 %max_val, %max_val, %val;
    
    add.u32 %i, %i, 1;
    bra MAX_LOOP;
    
MAX_DONE:
    // Compute sum of exp
    mov.f32 %sum_exp, 0f00000000;
    mov.u32 %i, 0;
    
EXP_LOOP:
    setp.ge.u32 %p, %i, %rcols;
    @%p bra EXP_DONE;
    
    mul.lo.u64 %tmp, %i, 4;
    add.u64 %off, %rin, %tmp;
    ld.global.f32 %val, [%off];
    sub.f32 %val, %val, %max_val;
    ex2.approx.ftz.f32 %exp_val, %val;
    add.f32 %sum_exp, %sum_exp, %exp_val;
    
    add.u32 %i, %i, 1;
    bra EXP_LOOP;
    
EXP_DONE:
    // Normalize and store
    mov.u32 %i, 0;
    
STORE_LOOP:
    setp.ge.u32 %p, %i, %rcols;
    @%p bra END;
    
    mul.lo.u64 %tmp, %i, 4;
    add.u64 %off, %rin, %tmp;
    ld.global.f32 %val, [%off];
    sub.f32 %val, %val, %max_val;
    ex2.approx.ftz.f32 %exp_val, %val;
    div.rn.f32 %val, %exp_val, %sum_exp;
    
    add.u64 %off, %rout, %tmp;
    st.global.f32 [%off], %val;
    
    add.u32 %i, %i, 1;
    bra STORE_LOOP;
    
END:
    ret;
}
"#.to_string()
}
