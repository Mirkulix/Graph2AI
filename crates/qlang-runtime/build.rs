fn main() {
    // Link Apple Accelerate framework on macOS for BLAS (cblas_sgemm).
    // This gives us hardware-optimized matrix multiply on M1/M2/M3 via NEON + AMX.
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
