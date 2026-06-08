// Build script to compile Rust kernels to PTX using cargo-oxide
//
// This script:
// 1. Checks if cargo-oxide is available
// 2. Compiles kernel modules to PTX
// 3. Embeds PTX in binary

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
    
    // Check for cargo-oxide
    let oxide_check = Command::new("cargo")
        .args(["+nightly-2026-04-03", "oxide", "--version"])
        .output();
    
    match oxide_check {
        Ok(output) if output.status.success() => {
            println!("cargo:warning=Found cargo-oxide, compiling kernels...");
            compile_kernels(&out_dir);
        }
        _ => {
            println!("cargo:warning=cargo-oxide not found, using stub kernels");
            generate_stub_ptx(&out_dir);
        }
    }
    
    // Tell cargo to rerun if kernel sources change
    println!("cargo:rerun-if-changed=src/kernels/");
}

fn compile_kernels(out_dir: &PathBuf) {
    // Compile each kernel module
    let kernel_modules = vec![
        ("elementwise", "src/kernels/elementwise.rs"),
        ("matmul", "src/kernels/matmul.rs"),
        ("attention", "src/kernels/attention.rs"),
    ];
    
    for (name, _path) in kernel_modules {
        let ptx_path = out_dir.join(format!("{}.ptx", name));
        
        // In real implementation:
        // cargo oxide build --release -p mistralrs-kernels --features cuda
        // and extract PTX
        
        // For now, generate stub PTX
        generate_stub_ptx_for_module(&ptx_path, name);
    }
}

fn generate_stub_ptx(out_dir: &PathBuf) {
    // Generate minimal PTX that can be loaded
    let ptx_path = out_dir.join("kernels.ptx");
    generate_stub_ptx_for_module(&ptx_path, "kernels");
}

fn generate_stub_ptx_for_module(ptx_path: &PathBuf, _name: &str) {
    let ptx = r#"
.version 8.0
.target sm_86
.address_size 64

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

.entry vecmul_f32(
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
    
    // Load A[row, i]
    mul.lo.u32 %tmp, %row, %rk;
    add.u32 %tmp, %tmp, %i;
    mul.lo.u64 %off_a, %tmp, 4;
    add.u64 %off_a, %ra, %off_a;
    ld.global.f32 %va, [%off_a];
    
    // Load B[i, col]
    mul.lo.u32 %tmp, %i, %rn;
    add.u32 %tmp, %tmp, %col;
    mul.lo.u64 %off_b, %tmp, 4;
    add.u64 %off_b, %rb, %off_b;
    ld.global.f32 %vb, [%off_b];
    
    // Multiply and accumulate
    mul.f32 %tmp2, %va, %vb;
    add.f32 %sum, %sum, %tmp2;
    
    add.u32 %i, %i, 1;
    bra LOOP;
    
LOOP_END:
    // Store C[row, col]
    mul.lo.u32 %tmp, %row, %rn;
    add.u32 %tmp, %tmp, %col;
    mul.lo.u64 %off_c, %tmp, 4;
    add.u64 %off_c, %rc, %off_c;
    st.global.f32 [%off_c], %sum;
    
END:
    ret;
}
"#;

    std::fs::write(ptx_path, ptx).expect("Failed to write PTX file");
    
    println!("cargo:warning=Generated stub PTX at {:?}", ptx_path);
}
