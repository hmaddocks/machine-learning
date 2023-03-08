use anyhow::anyhow;
use ndarray::{array, stack, Array1, Array2, Axis};
use std::fs::File;
use std::io::{BufRead, BufReader};

fn predict(input: f64, weights: &Array1<f64>) -> Array1<f64> {
    weights * input
}
fn predict_ndarray(inputs: &Array2<f64>, weights: &Array1<f64>) -> Array1<f64> {
    inputs.dot(weights)
}

fn loss(inputs: &Array2<f64>, labels: &Array1<f64>, weights: &Array1<f64>) -> f64 {
    let y_hat = predict_ndarray(inputs, weights);
    (&y_hat - labels).mapv(|x| x * x).sum() / (2.0 * inputs.shape()[0] as f64)
}

fn gradient(inputs: &Array2<f64>, labels: &Array1<f64>, weights: &Array1<f64>) -> Array1<f64> {
    let y_hat = predict_ndarray(inputs, weights);
    let diff = &y_hat - labels;
    inputs.t().dot(&diff) / inputs.shape()[0] as f64
}

fn train(inputs: &Array2<f64>, labels: &Array1<f64>, iterations: usize, lr: f64) -> Array1<f64> {
    let mut weights = Array1::zeros(inputs.shape()[1]);
    for i in 0..iterations {
        // println!(
        //     "Iteration {:4} => Loss: {:.20}",
        //     i,
        //     loss(inputs, labels, &weights)
        // );
        weights = &weights - gradient(inputs, labels, &weights) * lr;
    }
    weights
}

fn read_file(path: &str) -> anyhow::Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut col1 = Vec::new();
    let mut col2 = Vec::new();
    let mut col3 = Vec::new();
    let mut col4 = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() < 4 {
            return Err(anyhow!("Invalid file format"));
        }

        let num1 = parts[0].parse()?;
        let num2 = parts[1].parse()?;
        let num3 = parts[2].parse()?;
        let num4 = parts[3].parse()?;

        col1.push(num1);
        col2.push(num2);
        col3.push(num3);
        col4.push(num4);
    }

    let arr1 = Array1::from(col1);
    let arr2 = Array1::from(col2);
    let arr3 = Array1::from(col3);
    let arr4 = Array1::from(col4);

    Ok((arr1, arr2, arr3, arr4))
}

fn main() {
    let path = "pizza_3_vars.txt";
    match read_file(path) {
        Ok((inputs_1, inputs_2, inputs_3, labels)) => {
            let inputs = stack![
                Axis(1),
                Array1::ones(inputs_1.len()),
                inputs_1,
                inputs_2,
                inputs_3
            ];
            let weights = train(&inputs, &labels, 20000, 0.001);
            dbg!(&weights);

            for i in 0..5 {
                let prediction = predict(labels[i], &weights);
                println!("{i} -> {prediction} (label: {})", labels[i]);
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict() {
        let input = 3.;
        let weights = array![4., 3., 2.];

        assert_eq!(predict(input, &weights), array![12., 9., 6.]);
    }

    #[test]
    fn test_predict_zero_input() {
        let input = 0.0;
        let weights = array![1.0, 2.0, 3.0];
        assert_eq!(predict(input, &weights), array![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_predict_empty_weights() {
        let input = 2.0;
        let weights = array![];
        assert_eq!(predict(input, &weights), array![]);
    }

    #[test]
    fn test_predict_ndarray() {
        let inputs = array![[1., 1., 1.], [2., 2., 2.], [3., 3., 3.], [4., 4., 4.]];
        let weights = array![4., 3., 2.];

        assert_eq!(
            predict_ndarray(&inputs, &weights),
            array![9., 18., 27., 36.]
        );
    }

    // #[test]
    // fn test_predict_ndarray_empty_inputs() {
    //     let inputs = array![];
    //     let weights = array![1.0, 2.0, 3.0];
    //     assert_eq!(predict_ndarray(&inputs, &weights), array![]);
    // }

    #[test]
    #[should_panic]
    fn test_predict_ndarray_incompatible_dimensions() {
        let inputs = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let weights = array![1.0, 2.0];
        let _output = predict_ndarray(&inputs, &weights);
    }

    #[test]
    fn test_gradient() {
        let inputs = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]];
        let labels = array![3.0, 6.0, 9.0, 12.0];
        let weights = array![0.0, 0.0];
        let expected_output = array![5.0, 10.0];
        let output = gradient(&inputs, &labels, &weights);
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_gradient_all_zero() {
        let inputs = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]];
        let labels = array![3.0, 6.0, 9.0, 12.0];
        let weights = array![0.0, 0.0];
        let output = gradient(&inputs, &labels, &weights);
        assert!(output.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_gradient_one_input() {
        let inputs = array![[1.0, 2.0]];
        let labels = array![3.0];
        let weights = array![0.0, 0.0];
        let expected_output = array![3.0, 6.0];
        let output = gradient(&inputs, &labels, &weights);
        assert_eq!(output, expected_output);
    }

    #[test]
    #[should_panic]
    fn test_gradient_incompatible_shapes() {
        let inputs = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let labels = array![3.0, 6.0];
        let weights = array![0.0, 0.0, 0.0];
        let _output = gradient(&inputs, &labels, &weights);
    }

    #[test]
    fn test_train() {
        let inputs = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]];
        let labels = array![3.0, 6.0, 9.0, 12.0];
        let iterations = 1000;
        let lr = 0.01;
        let expected_weights = array![2.999999999999705, 0.999999999999903];
        let output_weights = train(&inputs, &labels, iterations, lr);
        assert_eq!(output_weights, expected_weights);
    }

    #[test]
    fn test_train_all_zero() {
        let inputs = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]];
        let labels = array![3.0, 6.0, 9.0, 12.0];
        let iterations = 1000;
        let lr = 0.01;
        let expected_weights = Array1::zeros(2);
        let output_weights = train(&inputs, &labels, iterations, lr);
        assert_eq!(output_weights, expected_weights);
    }

    #[test]
    fn test_train_one_input() {
        let inputs = array![[1.0, 2.0]];
        let labels = array![3.0];
        let iterations = 1000;
        let lr = 0.01;
        let expected_weights = array![2.9999999999996523, 0.999999999999857];
        let output_weights = train(&inputs, &labels, iterations, lr);
        assert_eq!(output_weights, expected_weights);
    }
}
