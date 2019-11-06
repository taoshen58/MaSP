import tensorflow as tf
import logging

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO, datefmt='%m/%d %I:%M:%S %p')


def log_specific_params(scope=None):
    # log the number of parameters
    logging.info("="*30)
    scope = scope or tf.get_variable_scope().name
    logging.info("In {}:".format(scope))
    tvars = tf.trainable_variables(scope)
    all_params_num = 0
    for elem in tvars:
        params_num = 1
        for l in elem.get_shape().as_list():
            params_num *= l
        logging.info("    {}: {}".format(elem.op.name, params_num))
        all_params_num += params_num
    logging.info('Trainable Parameters Number: %d' % all_params_num)
    logging.info("=" * 30)


if __name__ == '__main__':
    with tf.variable_scope("test"):
        a = tf.get_variable("a", [2, 3])
        b = tf.get_variable("b", [3, 4])
        log_specific_params()
    c = tf.get_variable("c", [3, 4])


