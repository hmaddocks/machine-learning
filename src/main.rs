use anyhow::anyhow;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn predict(input: f64, weight: f64, bias: f64) -> f64 {
    input * weight + bias
}

fn predict_vec(inputs: &[f64], weight: f64, bias: f64) -> Vec<f64> {
    inputs
        .iter()
        .map(|&input| predict(input, weight, bias))
        .collect()
}

fn average(samples: &Vec<f64>) -> f64 {
    samples.iter().sum::<f64>() / (samples.len() as f64)
}

fn loss(inputs: &[f64], labels: &[f64], weight: f64, bias: f64) -> f64 {
    let prediction = predict_vec(inputs, weight, bias);
    let error = prediction
        .iter()
        .zip(labels.iter())
        .map(|(l, r)| (l - r).powi(2))
        .collect();

    average(&error)
}

// def gradient(X, Y, w, b):
//     w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
//     b_gradient = 2 * np.average(predict(X, w, b) - Y)
//     return (w_gradient, b_gradient)

fn gradient(inputs: &[f64], labels: &[f64], weight: f64, bias: f64) -> (f64, f64) {
    let prediction = predict_vec(inputs, weight, bias);

    let weight_gradient = prediction
        .iter()
        .zip(labels.iter())
        .map(|(prediction, label)| prediction - label)
        .zip(inputs.iter())
        .map(|(error, input)| error * input)
        .collect();
    let weight_gradient = average(&weight_gradient) * 2.0;

    let bias_gradient = prediction
        .iter()
        .zip(labels.iter())
        .map(|(prediction, label)| prediction - label)
        .collect();
    let bias_gradient = average(&bias_gradient) * 2.0;
    (weight_gradient, bias_gradient)
}

fn train(inputs: Vec<f64>, labels: Vec<f64>, iterations: usize, learning_rate: f64) -> (f64, f64) {
    let mut weight = 0.0;
    let mut bias = 0.0;
    for _ in 0..iterations {
        let (weight_gradient, bias_gradient) = gradient(&inputs, &labels, weight, bias);
        weight -= weight_gradient * learning_rate;
        bias -= bias_gradient * learning_rate;
    }

    (weight, bias)
}

fn read_file(path: &str) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut inputs = Vec::new();
    let mut labels = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() < 2 {
            return Err(anyhow!("Invalid file format"));
        }

        let num1 = parts[0].parse()?;
        let num2 = parts[1].parse()?;

        inputs.push(num1);
        labels.push(num2);
    }

    Ok((inputs, labels))
}

fn main() {
    let path = "pizza.txt";
    match read_file(path) {
        Ok((inputs, labels)) => {
            let (weight, bias) = train(inputs, labels, 20000, 0.001);
            println!("weight: {:?}, bias: {:?}", weight, bias);
            let prediction = predict(20f64, weight, bias);
            println!("prediction: x = {:?}, y = {:?}", 20, prediction);
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
    fn test_predict_vec() {
        let (inputs, weight, bias) = (vec![20.0, 30.0], 1.5, 10.0);
        assert_eq!(predict_vec(&inputs, weight, bias), vec![40.0, 55.0]);
    }

    #[test]
    fn test_average() {
        let collection = vec![25.0, 35.0, 60.0];
        let average = average(&collection);
        assert_eq!(average, 40.0);
    }
}
