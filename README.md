<h1>Neural Network</h1>
<h3>About</h3>
<p>
    Light neural network library that can create, train, and test different models.
</p>
<h3>Usage</h3>
<h4>Classes</h4>
<ul>
    <li>
        <p>Model</p>
        <ul>
            <li>
                <p>Members</p>
                <ul>
                    <li>layers ~ all layers within current model object</li>
                </ul>
            </li>
            <li>
                <p>Methods</p>
                <ul>
                    <li>initialize(self, n, seed) ~ initializes all weights and biases within all layers. n is # of units in input.</li>
                    <li>forward_propagate(self, x_in) ~ takes a training input and propragate through all layers.</li>
                    <li>back_propagate(self, m, x_in, y_out, alpha) ~ performs backpropagation across all layers for a single epoch. y_out holds actual y values for current training set.</li>
                    <li>fit(self, X_train, Y_train, X_test, Y_test, alpha, epochs, seed, reg_rate) ~ trains data across all epochs. Returns training cost, test cost history, train accuracy, and test accuracy</li>
                    <li>predict(self, x_test) ~ returns forward_propagate results for x_test</li>
                    <li>summarize(self) ~ summarizes the model</li>
                </ul>
            </li>
        </ul>
    </li>
    <li>
        <p>Layer</p>
        <ul>
            <li>
                <p>Members</p>
                <ul>
                    <li>activation ~ holds activation function class for current layer</li>
                    <li>units ~ number of neurons in current layer</li>
                    <li>neurons ~ np.array holding activation values for each neuron</li>
                    <li>W_l ~ weight matrix for current layer</li>
                    <li>B_l ~ bias vector for current layer</li>
                </ul>
            </li>
            <li>
                <p>Methods</p>
                <ul>
                    <li>initialize(self, prev_layer_units) ~ initializes weights and biases within current layer.</li>
                    <li>feed_forward(self, a_in) ~ forward propagates through current layer. Weighted sum and activation calculations are performed here.</li>
                    <li>back_prop(self, del_J_z, prev_layer, alpha) ~ performs backprop through current layer. note, del_J_z represents partial derivative of cost W.R.T weighted sum, z.</li>
                    <li>summarize(self) ~ summarizes current layer</li>
                </ul>
            </li>
        </ul>
    </li>
</ul>
<h3>Example Usage</h3>
<img width="973" alt="Screen Shot 2023-07-29 at 12 47 20 AM" src="https://github.com/HenryChen4/Neural_Network/assets/71111859/b2bcf2e9-c703-47ed-9779-7dae30220d36">
