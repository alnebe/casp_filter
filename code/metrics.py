from tensorflow.keras.metrics import (FalseNegatives, FalsePositives,
                                      TrueNegatives, TruePositives,
                                      Recall, Precision)


class BaseMetrics:
    @staticmethod
    def recall(y_true, y_pred):
        r = Recall()
        r.update_state(y_true, y_pred)
        pre_res = r.result()
        res = pre_res.numpy()
        return res

    @staticmethod
    def precision(y_true, y_pred):
        p = Precision()
        p.update_state(y_true, y_pred)
        pre_res = p.result()
        res = pre_res.numpy()
        return res

    @staticmethod
    def tpr(y_true, y_pred):
        tp = TruePositives()
        tp.update_state(y_true, y_pred)
        pre_res = tp.result()
        tp_res = pre_res.numpy()

        fn = FalseNegatives()
        fn.update_state(y_true, y_pred)
        pre_res = fn.result()
        fn_res = pre_res.numpy()
        return tp_res / (tp_res + fn_res)

    @staticmethod
    def tnr(y_true, y_pred):
        tn = TrueNegatives()
        tn.update_state(y_true, y_pred)
        pre_res = tn.result()
        tn_res = pre_res.numpy()

        fp = FalsePositives()
        fp.update_state(y_true, y_pred)
        pre_res = fp.result()
        fp_res = pre_res.numpy()
        return tn_res / (tn_res + fp_res)


class AccMetrics:
    @staticmethod
    def fbeta(y_true, y_pred):
        # beta value is 0.5, means that recall weighs less than precision
        r = BaseMetrics.recall(y_true, y_pred)
        p = BaseMetrics.precision(y_true, y_pred)
        return (1 + 0.5 ** 2) * ((p * r) / ((0.5 ** 2 * p) + r))

    @staticmethod
    def f1(y_true, y_pred):
        r = BaseMetrics.recall(y_true, y_pred)
        p = BaseMetrics.precision(y_true, y_pred)
        return 2 * ((p * r) / (p + r))

    @staticmethod
    def bac(y_true, y_pred):
        tpr = BaseMetrics.tpr(y_true, y_pred)
        tnr = BaseMetrics.tnr(y_true, y_pred)
        return (tpr + tnr) / 2

    @staticmethod
    def acc(y_true, y_pred):
        tn = TrueNegatives()
        tn.update_state(y_true, y_pred)
        pre_res = tn.result()
        tn_res = pre_res.numpy()

        tp = TruePositives()
        tp.update_state(y_true, y_pred)
        pre_res = tp.result()
        tp_res = pre_res.numpy()

        fn = FalseNegatives()
        fn.update_state(y_true, y_pred)
        pre_res = fn.result()
        fn_res = pre_res.numpy()

        fp = FalsePositives()
        fp.update_state(y_true, y_pred)
        pre_res = fp.result()
        fp_res = pre_res.numpy()
        return (tp_res + tn_res) / (tp_res + tn_res + fp_res + fn_res)
