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

fn loss(inputs: &[f64], labels: &[f64], weight: f64, bias: f64) -> f64 {
    let length = inputs.len();
    let prediction = predict_vec(inputs, weight, bias);
    let error: f64 = prediction
        .iter()
        .zip(labels.iter())
        .map(|(l, r)| (l - r).powi(2))
        .sum();
    error / length as f64
}

fn train(
    inputs: Vec<f64>,
    labels: Vec<f64>,
    iterations: usize,
    lr: f64,
) -> anyhow::Result<(f64, f64)> {
    let mut weight = 0.0;
    let mut bias = 0.0;
    for _ in 0..iterations {
        let current_loss = loss(&inputs, &labels, weight, bias);
        if loss(&inputs, &labels, weight + lr, bias) < current_loss {
            weight += lr;
        } else if loss(&inputs, &labels, weight - lr, bias) < current_loss {
            weight -= lr
        } else if loss(&inputs, &labels, weight, bias + lr) < current_loss {
            bias += lr
        } else if loss(&inputs, &labels, weight, bias - lr) < current_loss {
            bias -= lr
        } else {
            return Ok((weight, bias));
        }
    }
    Err(anyhow!(
        "Couldn't converge within {} iterations",
        iterations
    ))
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
        Ok((inputs, labels)) => match train(inputs, labels, 10000, 0.01) {
            Ok((weight, bias)) => {
                println!("weight: {:?}, bias: {:?}", weight, bias);
                let prediction = predict(20f64, weight, bias);
                println!("prediction: x = {:?}, y = {:?}", 20, prediction);
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        },
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
}
