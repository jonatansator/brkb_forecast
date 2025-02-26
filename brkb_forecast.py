import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.graph_objects as go

class StockPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(StockPredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, 250)
        self.layer2 = nn.Linear(250, 100)
        self.layer3 = nn.Linear(100, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

# Step 1: add moving avg and bands
def calculate_metrics(price_data, days=50):
    dataset = []
    for i in range(days, len(price_data)):
        window = price_data[i - days:i]
        ma = np.mean(window)
        sd = np.std(window)
        low = ma - 2 * sd
        high = ma + 2 * sd
        dataset.append([price_data[i], ma, low, high])
    return pd.DataFrame(dataset, columns=['Price', 'MA', 'LB', 'UB'])

# Step 2: split data into train and test
def split_data(dataframe, prop=0.85):
    split_point = int(len(dataframe) * prop)
    train = dataframe[:split_point]
    test = dataframe[split_point:]
    return train, test

# Step 3: prep inputs and targets for training
def prepare_training_data(dataframe, window=30, output=7):
    inputs = []
    targets = []
    columns = dataframe.columns.tolist()
    for i in range(window, len(dataframe) - output + 1):
        window_data = dataframe[columns][i - window:i].values.T.flatten().tolist()
        target_data = dataframe['Price'][i:i + output].tolist()
        inputs.append(window_data)
        targets.append(target_data)
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

# Step 4: grab last window for predictions
def get_last_window(dataframe, window=30):
    print(f"Last window shape: {dataframe[-window:].values.shape}")
    columns = dataframe.columns.tolist()
    last_data = dataframe[columns][-window:].values.T.flatten().tolist()
    history = dataframe['Price'][-window:].tolist()
    return torch.tensor([last_data], dtype=torch.float32), history

# Step 5: load data and use last year's dataframe
data = pd.read_csv('BRKB.csv')
close = data['close'].values.tolist()[-252:]

# Step 6: set up params
epochs = 2000
window = 30
output = 7
learning_rate = 0.0001

# Step 7: process data with metrics
df = calculate_metrics(close)

# Step 8: split into train/test
train, test = split_data(df)
print(f"Test DataFrame shape: {test.shape}")
print(f"Test columns: {test.columns.tolist()}")

# Step 9: get training data ready
X, Y = prepare_training_data(train, window, output)

# Step 10: set up model 
model = StockPredictor(window * 4, output)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Step 11: train the model
for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print('Epochs left: ', epochs - epoch - 1)

# Step 12: get predictions
XX, history = get_last_window(test, window)
print(f"XX shape before prediction: {XX.shape}")
with torch.no_grad():
    test_outputs = model(XX)
predictions = test_outputs[0].tolist()

# Step 13: plot 
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=list(range(len(history))),
    y=history,
    mode='lines',
    name='Actual Price',
    line=dict(color='#FF6B6B', width=2),
    showlegend=True
))

last_actual = history[-1] if history else 0
forecasted_extended = [last_actual] + predictions
forecasted_x = list(range(len(history) - 1, len(history) + len(predictions)))

fig.add_trace(go.Scatter(
    x=forecasted_x,
    y=forecasted_extended,
    mode='lines',
    name='Forecasted Price',
    line=dict(color='#4ECDC4', width=2, dash='dash'),
    showlegend=True
))

fig.update_layout(
    plot_bgcolor='rgb(40, 40, 40)',
    paper_bgcolor='rgb(40, 40, 40)',
    font=dict(color='white'),
    xaxis=dict(
        title='Days',
        showgrid=True,
        gridcolor='rgba(255, 255, 255, 0.1)',
        gridwidth=0.5,
        griddash='dash',
        zeroline=False,
        showticklabels=True,
        color='white'
    ),
    yaxis=dict(
        title='Price (USD)',
        showgrid=True,
        gridcolor='rgba(255, 255, 255, 0.1)',
        gridwidth=0.5,
        griddash='dash',
        zeroline=False,
        showticklabels=True,
        color='white'
    ),
    showlegend=True,
    margin=dict(l=50, r=50, t=50, b=50),
    title='$BRKB Price Forecasting',
    title_font=dict(color='white')
)

fig.show()