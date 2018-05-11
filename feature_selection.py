import weka.core.jvm as jvm
import weka.core.converters as converters
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection


def all_feature(file):
    jvm.start(packages=True)
    data = converters.load_any_file(file)
    data.class_is_last()

    search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-T", "-1.7976931348623157E308", "-N", "-1"])
    attsel = AttributeSelection()
    attsel.search(search)

    evaluator = ASEvaluation(classname="weka.attributeSelection.ChiSquaredAttributeEval")
    attsel.evaluator(evaluator)
    attsel.select_attributes(data)
    t = attsel.ranked_attributes[:, 0]
    chi = t.astype(int)

    evaluator = ASEvaluation(classname="weka.attributeSelection.InfoGainAttributeEval")
    attsel.evaluator(evaluator)
    attsel.select_attributes(data)
    t = attsel.ranked_attributes[:, 0]
    info_gain = t.astype(int)

    evaluator = ASEvaluation(classname="weka.attributeSelection.GainRatioAttributeEval")
    attsel.evaluator(evaluator)
    attsel.select_attributes(data)
    t = attsel.ranked_attributes[:, 0]
    gain_ratio = t.astype(int)

    evaluator = ASEvaluation(classname="weka.attributeSelection.SymmetricalUncertAttributeEval")
    attsel.evaluator(evaluator)
    attsel.select_attributes(data)
    t = attsel.ranked_attributes[:, 0]
    symmetric_uncertainty = t.astype(int)

    jvm.stop()

    return chi, info_gain, gain_ratio, symmetric_uncertainty