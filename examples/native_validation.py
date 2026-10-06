"""Dense multiclass learning with validation loss and early stopping."""

import pybms as pb

df = pb.sample_data("iris")
split = pb.split_data(df, target="target", stratify=True)
# Reserve validation rows before learning scaling statistics.
training_table = split.X_train.assign(target=split.y_train)
inner = pb.split_data(training_table, target="target", stratify=True, seed=43)
scaled, scaler = pb.scale_data(inner.X_train, return_scaler=True)
X_val = scaler.transform(inner.X_test)
X_test = scaler.transform(split.X_test)
network = pb.build_network(input_dim=4, output_dim=3, task="multiclass", seed=42)
network.fit(
    scaled.to_numpy(),
    inner.y_train.to_numpy(),
    epochs=150,
    validation_data=(X_val, inner.y_test.to_numpy()),
    patience=10,
    verbose=0,
)
print(network.evaluate(X_test, split.y_test.to_numpy(), verbose=0))
print("Epochs:", len(network.history_["loss"]))
