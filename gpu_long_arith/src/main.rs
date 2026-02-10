use cudarc::driver::{LaunchConfig, PushKernelArg};

const MUL_SHADER: &'static str = include_str!("./shaders/mul.cu");
fn main() { 
    let ctx = cudarc::driver::CudaContext::new(0).expect("failed to init CUDA context");
    let stream = ctx.default_stream();

    let ptx = cudarc::nvrtc::compile_ptx(MUL_SHADER).expect("failed to compile shader");
    let module = ctx.load_module(ptx).expect("failed to load module");
    let inc_kernel = module.load_function("inc").expect("failed to load kernel");

    let input = stream.clone_htod(&[1; 100]).expect("failed to clone input from host to device");
    let mut out = stream.alloc_zeros::<u32>(100).expect("failed to allocate output");

    let mut builder = stream.launch_builder(&inc_kernel);
    builder.arg(&mut out);
    builder.arg(&input);
    unsafe {
        builder.launch(LaunchConfig::for_num_elems(100)).expect("failed to launch kernel");
    }

    let out_host: Vec<u32> = stream.clone_dtoh(&out).expect("failed to clone output from device to host");
    println!("Result: {out_host:?}");
}
