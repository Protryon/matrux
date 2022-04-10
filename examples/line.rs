use std::{collections::HashMap, time::Instant};

use matrux::{NeuralNetworkBuilder, activation, Matrix, optimizer::StochasticGradientDescent};


fn main() {
    let slope = 1.5;
    let base = 5.0;

    let line = |x: f64| {
        x * slope + base
    };

    const PT_COUNT: usize = 10;
    let points = (0..PT_COUNT).into_iter().map(|x| x as f64 - (PT_COUNT as f64  - 0.5)).map(|x| (x, line(x))).collect::<Vec<_>>();
    println!("{:#?}", points);

    let mut inputs = Matrix::new(2, points.len());
    let mut targets = Matrix::new(1, points.len());
    for (i, (input, output)) in points.into_iter().enumerate() {
        inputs[0][i] = input;
        inputs[1][i] = 1.0;
        targets[0][i] = output;
    }

    let mut network = NeuralNetworkBuilder::<f64>::new()
        .input(2)
        .add_dense_layer(3, activation::Linear)
        .add_dense_layer(1, activation::Linear);
    
    // let inputs = &[3.0, 1.0];
    // let mut outputs = network.eval(inputs);
    // let target = &[line(3.0)];
    // let mut outputs = Matrix::from_col(network.eval(inputs).into_iter());
    // let target = Matrix::from_col([line(3.0)]);

    // println!("f(3) = {}", outputs);

    let bp_plan = network.plan_backprop(PT_COUNT);
    // println!("{:#?}", bp_plan);

    let mut bp_inputs = HashMap::new();
    bp_inputs.insert("targets".to_string(), targets);
    bp_inputs.insert("inputs".to_string(), inputs);

    let mut optimizer = StochasticGradientDescent::new(0.001);

    let start = Instant::now();
    for _ in 0..10000 {
        network.fill_plan_weights(&mut bp_inputs);

        let (_, mut bp_outputs) = bp_plan.execute_cpu(&bp_inputs);
        let mut bp_vec = vec![];
        for i in 0..network.hidden_layers() {
            bp_vec.push(bp_outputs.remove(&*format!("gradient_{}", i)).unwrap());
        }
        network.apply_backprop(&mut optimizer, bp_vec);
        let outputs = bp_outputs.remove("outputs").unwrap();
        // outputs = Matrix::from_col(network.eval(inputs).into_iter());
        println!("f(3) = {}", outputs);
    }
    println!("elapsed = {} ms", start.elapsed().as_secs_f64() * 1000.0);

    // for _ in 0..1000 {
    //     let bp1 = network.calculate_backprop(inputs, &outputs[..], &target[..]);
    //     for line in &bp1 {
    //         println!("bp = \n{}", line);
    //     }
    //     network.apply_backprop(0.001, bp1);
    //     outputs = network.eval(inputs);
    //     println!("f(3) = {}", outputs.first().unwrap());
    // }

}