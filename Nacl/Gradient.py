import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import numpy as np
from tensorflow.python.ops.parallel_for.gradients import jacobian


class Custom_QNGDiff(tfq.differentiators.ParameterShift):

    ## Quantum Nautral Gradient based Differentiator..........
    def __init__(self, metric_tensor_fn):
      pass

    def prepGeometricTensor(self, state, vrs, n_params):
        
        #nparams = self.prep_variables(vrs)
        phi_r = tf.math.real(state)
        phi_c = tf.math.imag(state)
        jac_r = jacobian(phi_r, vrs)
        jac_c = jacobian(phi_c, vrs)
        if len(vrs) == 1:
            jac = tf.reshape(tf.complex(jac_r, jac_c), (state.shape + [n_params]))
            jac = tf.split(jac, n_params, axis=-1)
            jac = [tf.reshape(v, state.shape) for v in jac]
        else:
            jac = [tf.complex(jac_r[i], jac_c[i]) for i in range(n_params)]
        return jac

    def get_fisher_metric(self, state, variable):
        n_params = variable[0].shape
        jac = self.prepGeometricTensor(state, variable, n_params)
        variable = [variable]
        sf_metric =[]
        for i in range(n_params):
            for j in range(n_params):
                part_1 = tf.math.conj(tf.reshape(jac[i], (1, -1))) @ tf.reshape(jac[j], (-1, 1))
                part_2 = tf.math.conj(tf.reshape(jac[i], (1, -1))) @ tf.reshape(state, (-1, 1)) + \
                         tf.math.conj(tf.reshape(state, (1, -1))) @ tf.reshape(jac[j], (-1, 1))
                sf_metric.append(part_1 - part_2)
        eta = tf.math.real(tf.reshape(tf.stack(sf_metric), (n_params, n_params)))
        return eta
    
    @tf.function
    def differentiate_analytic(self, programs, symbol_names, symbol_values,
                               pauli_sums, forward_pass_vals, grad):
        n_symbols = tf.gather(tf.shape(symbol_names), 0)
        n_programs = tf.gather(tf.shape(programs), 0)
        n_ops = tf.gather(tf.shape(pauli_sums), 1)
        # Assume cirq.decompose() generates gates with at most two distinct
        # eigenvalues, which results in two parameter shifts.
        n_shifts = 2

        # STEP 1: Generate required inputs for executor
        # Deserialize programs and parse the whole parameterized gates
        # new_programs has [n_symbols, n_param_gates, n_shifts, n_programs].
        # These new_programs has programs that parameter-shift rule is applied,
        # so those programs has
        (new_programs, weights, shifts,
         n_param_gates) = tfq.differentiators.parameter_shift_util.parse_programs(
             programs, symbol_names, symbol_values, n_symbols)

        # Reshape & transpose new_programs, weights and shifts to fit into
        # the input format of tensorflow_quantum simulator.
        # [n_symbols, n_param_gates, n_shifts, n_programs]
        new_programs = tf.transpose(new_programs, [0, 2, 3, 1])
        weights = tf.transpose(weights, [0, 2, 3, 1])
        shifts = tf.transpose(shifts, [0, 2, 3, 1])

        # reshape everything to fit into expectation op correctly
        total_programs = n_programs * n_shifts * n_param_gates * n_symbols
        # tile up and then reshape to order programs correctly
        flat_programs = tf.reshape(new_programs, [total_programs])
        flat_shifts = tf.reshape(shifts, [total_programs])

        # tile up and then reshape to order ops correctly
        n_tile = n_shifts * n_param_gates * n_symbols
        flat_perturbations = tf.concat([
            tf.reshape(
                tf.tile(tf.expand_dims(symbol_values, 0),
                        tf.stack([n_tile, 1, 1])), [total_programs, n_symbols]),
            tf.expand_dims(flat_shifts, axis=1)
        ],
                                       axis=1)
        flat_ops = tf.reshape(
            tf.tile(tf.expand_dims(pauli_sums, 0), tf.stack([n_tile, 1, 1])),
            [total_programs, n_ops])
        # Append impurity symbol into symbol name
        new_symbol_names = tf.concat([
            symbol_names,
            tf.expand_dims(tf.constant(
                tfq.differentiators.parameter_shift_util._PARAMETER_IMPURITY_NAME),
                           axis=0)
        ],
                                     axis=0)

        # STEP 2: calculate the required expectation values
        expectations = self.expectation_op(flat_programs, new_symbol_names,
                                           flat_perturbations, flat_ops)

        # STEP 3: generate gradients according to the results

        # we know the rows are grouped according to which parameter
        # was perturbed, so reshape to reflect that
        grouped_expectations = tf.reshape(
            expectations,
            [n_symbols, n_shifts * n_programs * n_param_gates, -1])

        # now we can calculate the partial of the circuit output with
        # respect to each perturbed parameter
        def rearrange_expectations(grouped):

            def split_vertically(i):
                return tf.slice(grouped, [i * n_programs, 0],
                                [n_programs, n_ops])

            return tf.map_fn(split_vertically,
                             tf.range(n_param_gates * n_shifts),
                             dtype=tf.float32)

        # reshape so that expectations calculated on different programs are
        # separated by a dimension
        rearranged_expectations = tf.map_fn(rearrange_expectations,
                                            grouped_expectations)

        # now we will calculate all of the partial derivatives
        partials = tf.einsum(
            'spco,spc->sco', rearranged_expectations,
            tf.cast(
                tf.reshape(weights,
                           [n_symbols, n_param_gates * n_shifts, n_programs]),
                rearranged_expectations.dtype))

        # now apply the chain rule
        curr_final_grads =  tf.einsum('sco,co -> cs', partials, grad)
        sf_metric = self.get_fisher_metric(forward_pass_vals, symbol_values)
        natural_grads = tf.linalg.solve(sf_metric, tf.reshape(curr_final_grads, (-1, 1)))

        return natural_grads
    
    @tf.function
    def differentiate_sampled(self, programs, symbol_names, symbol_values,
                              pauli_sums, num_samples, forward_pass_vals, grad):
        n_symbols = tf.gather(tf.shape(symbol_names), 0)
        n_programs = tf.gather(tf.shape(programs), 0)
        n_ops = tf.gather(tf.shape(pauli_sums), 1)
        # Assume cirq.decompose() generates gates with at most two distinct
        # eigenvalues, which results in two parameter shifts.
        n_shifts = 2

        # STEP 1: Generate required inputs for executor
        # Deserialize programs and parse the whole parameterized gates
        # new_programs has [n_symbols, n_param_gates, n_shifts, n_programs].
        # These new_programs has programs that parameter-shift rule is applied,
        # so those programs has
        (new_programs, weights, shifts,
         n_param_gates) = tfq.differentiators.parameter_shift_util.parse_programs(
             programs, symbol_names, symbol_values, n_symbols)

        # Reshape & transpose new_programs, weights and shifts to fit into
        # the input format of tensorflow_quantum simulator.
        # [n_symbols, n_param_gates, n_shifts, n_programs]
        new_programs = tf.transpose(new_programs, [0, 2, 3, 1])
        weights = tf.transpose(weights, [0, 2, 3, 1])
        shifts = tf.transpose(shifts, [0, 2, 3, 1])

        # reshape everything to fit into expectation op correctly
        total_programs = n_programs * n_shifts * n_param_gates * n_symbols
        # tile up and then reshape to order programs correctly
        flat_programs = tf.reshape(new_programs, [total_programs])
        flat_shifts = tf.reshape(shifts, [total_programs])

        # tile up and then reshape to order ops correctly
        n_tile = n_shifts * n_param_gates * n_symbols
        flat_perturbations = tf.concat([
            tf.reshape(
                tf.tile(tf.expand_dims(symbol_values, 0),
                        tf.stack([n_tile, 1, 1])), [total_programs, n_symbols]),
            tf.expand_dims(flat_shifts, axis=1)
        ],
                                       axis=1)
        flat_ops = tf.reshape(
            tf.tile(tf.expand_dims(pauli_sums, 0), tf.stack([n_tile, 1, 1])),
            [total_programs, n_ops])
        flat_num_samples = tf.reshape(
            tf.tile(tf.expand_dims(num_samples, 0), tf.stack([n_tile, 1, 1])),
            [total_programs, n_ops])
        # Append impurity symbol into symbol name
        new_symbol_names = tf.concat([
            symbol_names,
            tf.expand_dims(tf.constant(
                tfq.differentiators.parameter_shift_util._PARAMETER_IMPURITY_NAME),
                           axis=0)
        ],
                                     axis=0)

        # STEP 2: calculate the required expectation values
        expectations = self.expectation_op(flat_programs, new_symbol_names,
                                           flat_perturbations, flat_ops,
                                           flat_num_samples)

        # STEP 3: generate gradients according to the results

        # we know the rows are grouped according to which parameter
        # was perturbed, so reshape to reflect that
        grouped_expectations = tf.reshape(
            expectations,
            [n_symbols, n_shifts * n_programs * n_param_gates, -1])

        # now we can calculate the partial of the circuit output with
        # respect to each perturbed parameter
        def rearrange_expectations(grouped):

            def split_vertically(i):
                return tf.slice(grouped, [i * n_programs, 0],
                                [n_programs, n_ops])

            return tf.map_fn(split_vertically,
                             tf.range(n_param_gates * n_shifts),
                             dtype=tf.float32)

        # reshape so that expectations calculated on different programs are
        # separated by a dimension
        rearranged_expectations = tf.map_fn(rearrange_expectations,
                                            grouped_expectations)

        # now we will calculate all of the partial derivatives
        partials = tf.einsum(
            'spco,spc->sco', rearranged_expectations,
            tf.cast(
                tf.reshape(weights,
                           [n_symbols, n_param_gates * n_shifts, n_programs]),
                rearranged_expectations.dtype))

        # now apply the chain rule
        curr_final_grads =  tf.einsum('sco,co -> cs', partials, grad)
        sf_metric = self.get_fisher_metric(forward_pass_vals, symbol_values)
        natural_grads = tf.linalg.solve(sf_metric, tf.reshape(curr_final_grads, (-1, 1)))

        return natural_grads
    





        
    
     
