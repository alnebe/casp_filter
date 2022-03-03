"""Decoy generation workflow"""

# Import relevant packages
from CGRtools.files import RDFRead, RDFWrite
from CGRtools.exceptions import *
from datetime import date
from tqdm import tqdm
from utils import (generate_reactions, remove_reagents,
                   containers_split, not_radical,
                   db_check, _save_log, _util_file,
                   time_func, _to_compare)

import pickle
import argparse



def main(TYPE: str, name_in: str,
         log_filename: str, name_out: str,
         verbose: bool, max_decoys: int,
         limit: int, log: bool, STOP: int):
    """
    Main decoys generation routine

    =======================   ===================================
    Reactions in input file   1 316 541
    Data                      reactions, type: ReactionContainer
    =======================   ===================================

    Parameters
    ----------
    :param log: bool. Logging if True.
    :param limit:
    :param max_decoys:
    :param verbose:
    :param name_out:
    :param log_filename:
    :param total:
    :param name_in:
    :param TYPE:
    :return: None
    """
    if TYPE == "CGR":
        word = "cgr"
    elif TYPE == "SMILES":
        word = "smiles"

    with open("comparison_set_{}_28022022_uspto.pickle".format(word), "rb") as pkl:
        comparison_set = pickle.load(pkl)

    with RDFRead(name_in, indexable=True) as input_rdf, \
            open(name_out, "a") as w, RDFWrite(w) as output_rdf, \
            open("NONRECON_{}.rdf".format(word), "a") as z, RDFWrite(z) as nonrecon_rdf, \
            open("REMOVED_{}.rdf".format(word), "a") as y, RDFWrite(y) as rmvd_rdf:

        signatures = {}
        conflict = set()

        for n, reaction in tqdm(enumerate(input_rdf), total=len(input_rdf)):
            if STOP:
                if n == STOP:
                    print("The set limit has been reached. Reactions processed: {}".format(n))
                    break

            comparison_doc = {}
            reaction = remove_reagents(reaction)
            if reaction is not None:
                reaction = containers_split(reaction)
                cgr = reaction.compose()
                if not_radical(cgr):
                    if db_check(cgr):  # the num of dyn bonds in a CGR must be > 1 and < or = 7

                        doc = generate_reactions(reaction, max_decoys, limit)  # contain ReactionContainer's

                        for rxn in doc:  # in the set of generated reactions, we are looking for the initial
                            if _to_compare(TYPE, rxn) in comparison_set:  # comparing str by TYPE: CGR или SMILES
                                rxn.meta["type"] = "Reconstructed"
                            else:  # this reaction is not in the original USPTO dataset
                                rxn.meta["type"] = "Decoy"
                            comparison_doc.update({_to_compare(TYPE, rxn): rxn})  # duplicate Dictionary

                        intersection = set(comparison_doc).intersection(set(signatures))

                        for duplicate in intersection:  # looking for duplicate decoys
                            if comparison_doc[duplicate].meta["type"] != signatures[duplicate]:
                                conflict.add(duplicate)
                            else:
                                del comparison_doc[duplicate]

                        # We check if among the generated reactions there is the original one,
                        # if not, then we discard the entire set:
                        if any([True if _to_compare(TYPE, rxn) == _to_compare(TYPE, reaction) else False for rxn in
                                doc]):
                            if verbose:
                                print("REC;ID:{};TIME:{};NUM:{}\n".format(
                                    reaction.meta["Reaction_ID"],
                                    generate_reactions.elapsed,
                                    len(comparison_doc)))
                            if log:
                                _save_log(log_filename,
                                          str("REC;ID:{};TIME:{};NUM:{}\n".format(
                                              reaction.meta["Reaction_ID"],
                                              generate_reactions.elapsed,
                                              len(comparison_doc))))
                            for rxn in doc:
                                if _to_compare(TYPE, rxn) in comparison_doc:  # write if reaction str is not removed
                                    output_rdf.write(rxn)
                                    signatures.update({_to_compare(TYPE,
                                                                   rxn): rxn})  # write str to sign.(to rm dupl., later)
                        else:  # the original reaction was not reconstructed
                            if verbose:
                                print("NREC;ID:{};TIME:{};NUM:{}\n".format(
                                    reaction.meta["Reaction_ID"],
                                    generate_reactions.elapsed,
                                    len(comparison_doc)))
                            if log:
                                _save_log(log_filename,
                                          str("NREC;ID:{};TIME:{};NUM:{}\n".format(
                                              reaction.meta["Reaction_ID"],
                                              generate_reactions.elapsed,
                                              len(comparison_doc))))
                            for rxn in doc:
                                if _to_compare(TYPE, rxn) in comparison_doc:  # write if reaction str is not removed
                                    nonrecon_rdf.write(rxn)
                                    # The orig. reaction was not restored, however, among
                                    # the generated reactions there may be a certain number of
                                    # reactions with the "Reconstructed" type, so as not to
                                    # delete them as duplicates, we will not write all discarded reactions in sign.
                    else:
                        reaction.meta.update({"removed": "dynbonds"})
                        rmvd_rdf.write(reaction)
                else:
                    reaction.meta.update({"removed": "radical"})
                    rmvd_rdf.write(reaction)
    return conflict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('TYPE', type=str,
                        help='Determines what kind of reaction will be converted')
    parser.add_argument('name_in', type=str,
                        help='Path to the main rdf file')
    parser.add_argument('-log_filename', type=str, default='GENERATING_LOG_{}.rdf'.format(date.today()),
                        help='Output rdf filename; defaults to: GENERATING_LOG_{current date}')
    parser.add_argument('-name_out', type=str, default='DECOYS_FROM_{}.rdf'.format(date.today()),
                        help='Output rdf filename; defaults to: DECOYS_FROM_{current date}')
    parser.add_argument('-verbose', type=bool, default=False,
                        help='Verbose printing; defaults to False')
    parser.add_argument('-max_decoys', '--num', type=int, default=50,
                        help='The maximum number of decoys that can be generated; defaults to 50')
    parser.add_argument('-limit', '--lim', type=int, default=10,
                        help='The maximum number of template applying; defaults to 10')
    parser.add_argument('--log', type=bool, default=True,
                        help='Whether to log wall times / generated num of rxns / etc., default True')
    parser.add_argument('--STOP', type=int, default=0,
                        help='Where to stop; defaults to 0, mean all')
    args = parser.parse_args()
    verbose = bool(args.verbose)
    log = bool(args.log)

    _util_file("NONRECON.rdf")  # deleting the files if it was created earlier in the directory
    _util_file("REMOVED.rdf")
    _util_file("CONFLICT.pickle")
    _util_file(str(args.name_out))
    if log:
        _util_file(str(args.log_filename))

    with open("Config.pickle", "wb") as config:
        config_list = [
            str(args.TYPE),
            str(args.name_in),
            str(args.log_filename),
            str(args.name_out),
            bool(args.verbose),
            int(args.max_decoys),
            int(args.limit),
            bool(args.log),
            int(args.STOP)
        ]
        pickle.dump(config_list, config)

    conflict = main(str(args.TYPE), str(args.name_in), str(args.log_filename), str(args.name_out), bool(args.verbose),
                    int(args.max_decoys), int(args.limit), bool(args.log), int(args.STOP))

    with open("CONFLICT.pickle", "wb") as pkl:  # a set of reactions for which the real type is reconstructed
        pickle.dump(conflict, pkl)
