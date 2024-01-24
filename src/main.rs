use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new("mse_loss_chart.png", (640, 480)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .caption("MSE Loss over Epochs", ("sans-serif", 40).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..1000, 0f64..1f64)?;

    chart.configure_mesh().draw()?;


    let inputs = vec![vec![6.2, 0.3], vec![6.7, 0.14], vec![7.6, 0.4], vec![8.9, 0.31], vec![9.1, 0.68]]; // Example inputs
    let outputs = vec![0.0, 0.0, 0.0, 1.0, 1.0]; // Example outputs (labels)

    let mut weights = vec![0.0, 0.0]; // Initialize weights
    let mut bias = 0.0; // Initialize bias
    let learning_rate = 0.1;
    let epochs = 1000;

    let mut epoch_losses = Vec::new();

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (input, &output) in inputs.iter().zip(outputs.iter()) {
            // Forward pass
            let weighted_sum = input.iter().zip(weights.iter()).map(|(x, w)| x * w).sum::<f64>() + bias;
            let prediction = sigmoid(weighted_sum);
            
            // loss
            total_loss += mean_squared_error(output, prediction);
            
            // Backward pass (Gradient Descent)
            let error = prediction - output;
            for (j, input_val) in input.iter().enumerate() {
                weights[j] -= learning_rate * error * input_val * sigmoid(weighted_sum) * (1.0 - sigmoid(weighted_sum));
            }
            bias -= learning_rate * error * sigmoid(weighted_sum) * (1.0 - sigmoid(weighted_sum));
        }
        epoch_losses.push((epoch, total_loss / inputs.len() as f64));
    }

    println!("Trained weights: {:?}", weights);
    println!("Trained bias: {:?}", bias);

    chart
        .draw_series(LineSeries::new(epoch_losses, &RED))?
        .label("MSE Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn mean_squared_error(y_true: f64, y_pred: f64) -> f64 {
    (y_true - y_pred).powi(2)
}