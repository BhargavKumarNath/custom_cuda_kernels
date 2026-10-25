//! Compiles every `csrc/kernels/*.cu` file with `nvcc` and links the result
//! into this crate's `cdylib`.
//!
//! On the `msvc` target, `nvcc` shells out to `cl.exe` as its host compiler
//! and `cl.exe` is frequently not on `PATH` (Visual Studio Build Tools are
//! normally only put on `PATH` inside a "Developer Command Prompt"). Rather
//! than requiring that environment, this script uses the `cc` crate's MSVC
//! auto-detection (the same mechanism `cargo build` itself relies on for C
//! dependencies) to locate `cl.exe` and its required `INCLUDE`/`LIB`/`PATH`
//! environment, then passes that through to `nvcc` via `-ccbin` plus the
//! subprocess environment.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");

    let kernels_dir = PathBuf::from("csrc/kernels");
    let cu_files: Vec<PathBuf> = std::fs::read_dir(&kernels_dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|ext| ext == "cu"))
                .collect()
        })
        .unwrap_or_default();

    if cu_files.is_empty() {
        // No kernels written yet — nothing to compile.
        return;
    }

    let cuda_path = locate_cuda();
    let nvcc_name = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
    let nvcc = cuda_path.join("bin").join(nvcc_name);
    if !nvcc.exists() {
        panic!(
            "nvcc not found at {}. Set CUDA_PATH/CUDA_HOME to the CUDA toolkit install directory.",
            nvcc.display()
        );
    }

    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_string());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let includes_dir = PathBuf::from("csrc/includes");

    let is_msvc = env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    let msvc_tool = is_msvc.then(|| cc::Build::new().get_compiler());

    let mut objects = Vec::new();
    for cu in &cu_files {
        println!("cargo:rerun-if-changed={}", cu.display());

        let stem = cu.file_stem().unwrap().to_string_lossy().to_string();
        let ext = if is_msvc { "obj" } else { "o" };
        let obj = out_dir.join(format!("{stem}.{ext}"));

        let mut cmd = Command::new(&nvcc);
        cmd.arg("-c")
            .arg(cu)
            .arg("-o")
            .arg(&obj)
            .arg("-O3")
            .arg("--use_fast_math")
            .arg(format!("-arch={arch}"))
            .arg("-I")
            .arg(&includes_dir)
            .arg("-std=c++17");

        if let Some(tool) = &msvc_tool {
            cmd.arg("-ccbin").arg(tool.path());
            for (key, value) in tool.env() {
                cmd.env(key, value);
            }
        } else {
            // Newer glibc (2.39+, e.g. Ubuntu 24.04+/26.04) declares rsqrt/rsqrtf
            // (and a few other IEC 60559 extension functions) under
            // _GNU_SOURCE with a `noexcept` exception spec that conflicts
            // with CUDA's own crt/math_functions.h declarations of the same
            // symbols, causing nvcc's host-compiler pass to fail with
            // "exception specification is incompatible". _DEFAULT_SOURCE
            // still gives glibc's usual extensions without pulling in the
            // IEC 60559 extension set that collides with CUDA's headers.
            cmd.arg("-Xcompiler").arg("-U_GNU_SOURCE");
            cmd.arg("-Xcompiler").arg("-D_DEFAULT_SOURCE");
        }

        let status = cmd
            .status()
            .unwrap_or_else(|e| panic!("failed to launch nvcc for {}: {e}", cu.display()));
        if !status.success() {
            panic!("nvcc failed compiling {} (exit: {status})", cu.display());
        }
        objects.push(obj);
    }

    // Archive the compiled kernel objects into a static lib and let `cc`
    // emit the correct cargo:rustc-link-* directives.
    let mut build = cc::Build::new();
    for obj in &objects {
        build.object(obj);
    }
    build.compile("cuda_kernels");

    let cuda_lib_dir = if cfg!(windows) {
        cuda_path.join("lib").join("x64")
    } else {
        cuda_path.join("lib64")
    };
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=cudart");

    // The nvcc-compiled kernel objects are emitted as C++ TUs (exception
    // tables, `__gxx_personality_v0`, etc. even though the kernels
    // themselves don't throw), but `cc::Build::compile` here only archives
    // pre-built objects, so it never links libstdc++ the way it would for a
    // crate it compiles from C++ sources itself. Without this, linking the
    // final cdylib fails / produces an extension module with an unresolved
    // `__gxx_personality_v0` symbol at import time.
    if cfg!(all(unix, not(target_os = "macos"))) {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
    }
}

fn locate_cuda() -> PathBuf {
    if let Ok(p) = env::var("CUDA_PATH") {
        return PathBuf::from(p);
    }
    if let Ok(p) = env::var("CUDA_HOME") {
        return PathBuf::from(p);
    }
    panic!("CUDA_PATH or CUDA_HOME must be set to the CUDA toolkit install directory");
}
