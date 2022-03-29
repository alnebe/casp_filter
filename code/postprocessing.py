import pickle
import argparse

from CGRtools import RDFRead
from StructureFingerprint import LinearFingerprint, MorganFingerprint


def main(input_rdf: str, out_fp_filename: str, fp_type: str = "Linear",
         min_rad: int = 2, max_rad: int = 6, bit_length: int = 4096, n_bit_pairs: int = 4):

    if fp_type == "Linear":
        fp = LinearFingerprint(min_radius=min_rad,
                               max_radius=max_rad,
                               length=bit_length,
                               number_bit_pairs=n_bit_pairs)
    elif fp_type == "Morgan":
        fp = MorganFingerprint(min_radius=min_rad,
                               max_radius=max_rad,
                               length=bit_length,
                               include_hydrogens=False)

    with RDFRead(input_rdf, indexable=True) as rdf, \
            open(f"fps_pkls/{out_fp_filename}", "ab") as fp_pkl, \
            open("log_bad.txt", "a") as log:

        num = 0
        desc_dict = {}

        for n, rxn in enumerate(rdf, start=1):
            try:
                num += 1
                desc_dict.update({str(num): [fp.transform([rxn.compose()]),
                                             1 if rxn.meta["type"].startswith("Reconstructed") else 0]})
            except Exception as e:
                log.write(f"ERROR:'{e}';ENUM:{n}")

        pickle.dump(desc_dict, fp_pkl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name_in', type=str,
                        help='Path to the main rdf file')
    parser.add_argument('name_out', type=str,
                        help='Name of the output pickle file with fingerprints')
    parser.add_argument('-fp_type', type=str, default="Linear",
                        help='Type of the output fingerprints, default to Linear; another - Morgan')
    parser.add_argument('-min_rad', type=int, default=2,
                        help='Minimal length of fragment')
    parser.add_argument('-max_rad', type=int, default=6,
                        help='Maximal length of fragment')
    parser.add_argument('-bit_length', type=int, default=4096,
                        help='Length of bit string')
    parser.add_argument('-n_bit_pairs', type=int, default=4,
                        help="""describe how much repeating fragments we can count in hashable fingerprint (if number of
                         fragment in molecule greater or equal this number, we will be activate only this number
                         of fragments""")
    args = parser.parse_args()

    main(str(args.name_in), str(args.name_out), str(args.fp_type), int(args.min_rad),
         int(args.max_rad), int(args.bit_length), int(args.n_bit_pairs))
