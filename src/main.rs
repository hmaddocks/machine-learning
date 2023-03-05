use anyhow::anyhow;
use ndarray::{array, Array1};
use std::fs::File;
use std::io::{BufRead, BufReader};

fn predict(input: f64, weight: f64, bias: f64) -> f64 {
    input * weight + bias
}

fn predict_ndarray(inputs: &Array1<f64>, weight: f64, bias: f64) -> Array1<f64> {
    inputs
        .iter()
        .map(|&input| predict(input, weight, bias))
        .collect()
}

fn average(samples: &Array1<f64>) -> f64 {
    samples.iter().sum::<f64>() / (samples.len() as f64)
}

fn loss(inputs: &Array1<f64>, labels: &Array1<f64>, weight: f64, bias: f64) -> f64 {
    let prediction = predict_ndarray(inputs, weight, bias);
    let error = prediction
        .iter()
        .zip(labels.iter())
        .map(|(l, r)| (l - r).powi(2))
        .collect();

    average(&error)
}

fn gradient(inputs: &Array1<f64>, labels: &Array1<f64>, weight: f64, bias: f64) -> (f64, f64) {
    let prediction = predict_ndarray(inputs, weight, bias);

    let errors: Array1<f64> = prediction
        .iter()
        .zip(labels.iter())
        .map(|(prediction, label)| prediction - label)
        .collect();

    let weight_gradient = errors
        .iter()
        .zip(inputs.iter())
        .map(|(error, input)| error * input)
        .collect();

    let weight_gradient = average(&weight_gradient) * 2.0;
    let bias_gradient = average(&errors) * 2.0;
    (weight_gradient, bias_gradient)
}

fn train(
    inputs: Array1<f64>,
    labels: Array1<f64>,
    iterations: usize,
    learning_rate: f64,
) -> (f64, f64) {
    let mut weight = 0.0;
    let mut bias = 0.0;
    for _ in 0..iterations {
        let (weight_gradient, bias_gradient) = gradient(&inputs, &labels, weight, bias);
        weight -= weight_gradient * learning_rate;
        bias -= bias_gradient * learning_rate;
    }

    (weight, bias)
}

fn read_file(path: &str) -> anyhow::Result<(Array1<f64>, Array1<f64>)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut col1 = Vec::new();
    let mut col2 = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() < 2 {
            return Err(anyhow!("Invalid file format"));
        }

        let num1 = parts[0].parse()?;
        let num2 = parts[1].parse()?;

        col1.push(num1);
        col2.push(num2);
    }

    let arr1 = Array1::from(col1);
    let arr2 = Array1::from(col2);

    Ok((arr1, arr2))
}

fn main() {
    let path = "pizza.txt";
    match read_file(path) {
        Ok((inputs, labels)) => {
            let (weight, bias) = train(inputs, labels, 20000, 0.001);
            println!("weight: {:?}, bias: {:?}", weight, bias);

            let target = 13.;
            let prediction = predict(target, weight, bias);
            println!("prediction: x = {:?}, y = {:?}", target, prediction);
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
        let (input, weight, bias) = (20.0, 1.5, 10.0);
        assert_eq!(predict(input, weight, bias), 40.0);
    }

    #[test]
    fn test_predict_ndarray() {
        let (inputs, weight, bias) = (array![20.0, 30.0], 1.5, 10.0);
        assert_eq!(predict_ndarray(&inputs, weight, bias), array![40.0, 55.0]);
    }

    #[test]
    fn test_average() {
        let collection = array![25.0, 35.0, 60.0];
        let average = average(&collection);
        assert_eq!(average, 40.0);
    }
}
