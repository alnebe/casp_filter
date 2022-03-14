"""Some routines for generation workflow"""
from CGRtools.reactor import Reactor
from CGRtools.containers import ReactionContainer
from CGRtools.exceptions import InvalidAromaticRing, MappingError
from template_utils import (CC_append_filter, CC_break_filter,
                            num_atoms_filter, num_dynbonds_filter,
                            prod_to_reac_filter, reac_to_prod_filter)

import itertools
import time
import os


def time_func(function):
    """
    This decorator calculates the amount of time a function takes to execute
    :param function: function to time evaluate
    """

    def new_func(*args, **kwargs):
        start = time.time()
        function_result = function(*args, **kwargs)
        new_func.elapsed = round(time.time() - start, 3)
        return function_result

    return new_func


def _to_compare(TYPE: str, reaction: ReactionContainer) -> str:
    """
    Transformation of the ReactionContainer into linear notation.
    Type determines what kind of reaction will be converted, for example:

    if TYPE == "CGR":
        return: 'C1CN(CCC1)[.>-]C([->.]Br)CCCl'
    elif TYPE == "SMILES":
        return: 'C(CCBr)Cl.C1CCCCN1>>C1CCCCN1CCCCl'

    :param TYPE: comparison type
    :param reaction: ReactionContainer
    :return: str
    """
    if TYPE == "CGR":
        return str(reaction.compose())
    elif TYPE == "SMILES":
        return str(reaction)


def _save_log(filename, string):
    """
    Logging function.
    :param filename: output log filename
    :param string: string to add to log
    :return: None
    """
    with open(filename, "a") as flog:
        flog.write(string)


def _util_file(filename):
    """
    File utilize function.
    :param filename: filename
    :return: None
    """
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


def remove_reagents(reaction):
    """
    Removing unchanging molecules, and checking reaction properties.
    :param reaction: input reaction
    :return: ReactionContainer or None in some cases
    """
    try:
        cgr = reaction.compose()
    except ValueError:
        return None

    to_remove = []
    for n, i in enumerate(cgr.split()):  # remove unchanging molecules
        if any([z[1].charge != z[1].p_charge for z in i.atoms()]) or \
                any([z[2].order != z[2].p_order for z in i.bonds()]):
            continue
        to_remove.extend([x for x, _ in i.atoms()])
    for mol in reaction.molecules():
        for z in to_remove:
            try:
                mol.delete_atom(z)
            except KeyError:
                continue

    try:
        reaction.canonicalize()
    except InvalidAromaticRing:
        return None
    reaction.flush_cache()
    reactants = [x for x in reaction.reactants if x]
    products = [x for x in reaction.products if x]
    if len(reactants) > 0 and len(products) > 0:
        reaction = ReactionContainer(reactants=reactants,
                                     products=products,
                                     meta=reaction.meta)
        return reaction
    else:
        return None


def db_check(cgr):
    if 1 < len(cgr.center_bonds) <= 7:
        return True
    else:
        return False


def number_of_atoms_check(rxn):
    num_atoms_rxn = sum([x.atoms_count for x in rxn.reactants]) + sum([x.atoms_count for x in rxn.products])
    if num_atoms_rxn > 150:
        return False
    else:
        return True


def not_radical(cgr):
    """
    Checking for charged atoms in a Condensed Graph of Reaction.
    :param cgr: Condensed Graph of the input reaction
    :return: bool
    """
    if cgr and cgr.center_atoms:
        if any(x.is_radical or x.p_is_radical for _, x in cgr.atoms()):
            return False
    return True


def containers_split(reaction):
    """
    Separation of MoleculeContainers.
    :param reaction: input reaction
    :return: ReactionContainer
    """
    new_reactants = [x for container in reaction.reactants for x in container.split()]
    new_products = [x for container in reaction.products for x in container.split()]
    return ReactionContainer(reactants=new_reactants,
                             products=new_products,
                             meta=reaction.meta)


@time_func
def generate_reactions(reaction: ReactionContainer,
                       max_decoys: int=50,
                       limit: int=10,
                       save_bad: bool=False):
    """
    Accumulation of generated reactions.

    time_func decorator used to estimate processing time.

    :param reaction: input reaction
    :param max_decoys: max number of reaction to generate
    :param limit: max number of reaction from one transformation and max number of error raise
    :return: set(ReactionContainer, ...)
    """
    doc = set()
    num_err = 0
    for rxn in apply_templates(reaction, save_bad, max_decoys):

        if num_err > limit:  # many MappingError's may occur
            break

        if len(doc) == max_decoys:  # maximum number of decoys generated
            break

        new_reaction = ReactionContainer(reactants=reaction.reactants,
                                         products=rxn.products,
                                         meta=rxn.meta)
        new_reaction.meta.update(reaction.meta)
        try:
            new_reaction = remove_reagents(new_reaction)
            if new_reaction is not None:
                new_reaction = containers_split(new_reaction)
                new_reaction.canonicalize()
            else:
                continue
        except InvalidAromaticRing:
            num_err += 1
            continue
        try:
            if (new_reaction.compose()).center_atoms:
                doc.add(new_reaction)
            else:
                continue
        except MappingError:
            num_err += 1
            continue
    return doc


def apply_templates(reaction: ReactionContainer,
                    save_bad: bool=False,
                    max_decoys: int=50):
    """
    Reaction generator.
    :param reaction: input reaction
    :param save_bad: save invalid rules (check generated reaction metadata for validation)
    :param max_decoys: here, this param. need to define the max. num. of iter. of the cycle (to get rid of the closure)
    :return: yield(ReactionContainer) of generated reaction
    """
    templates = get_templates(reaction, save_bad)

    if number_of_atoms_check(reaction):
        big_rxn = False
    else:
        big_rxn = True

    reactors = [Reactor(template,
                        delete_atoms=True,
                        one_shot=False,
                        automorphism_filter=False,
                        polymerise_limit=5) for template in templates]  # NB! CGRtools v. > 4.1.22
    queue = [r(reaction.reactants) for r in reactors]

    seen = set()
    num_try = 0
    while queue:
        reactor_call = queue.pop(0)
        try:
            for new_reaction in reactor_call:
                num_try += 1
                if num_try > max_decoys * 2:
                    queue = False
                    break
                if str(new_reaction) not in seen:
                    if big_rxn:
                        seen.add(str(new_reaction))
                        yield new_reaction
                    else:
                        if number_of_atoms_check(new_reaction):
                            seen.add(str(new_reaction))
                            yield new_reaction
        except (KeyError, IndexError):
            continue


def get_templates(reaction: ReactionContainer,
                  save_bad: bool=False):
    """
    Obtaining the templates of reaction transformations. NB! CGRtools.enumerate_centers() is used.
    :param save_bad: save invalid rules (check generated reaction metadata for validation)
    :param reaction: input reaction
    :return: list[ReactionContainer, ...]
    """

    rules = []

    for partial_reaction in itertools.islice(reaction.enumerate_centers(), 0, 11):  # getting single stages
        for reaction_center in partial_reaction.extended_centers_list:

            reactants = reaction.reactants
            products = reaction.products

            cleavage = set(partial_reaction.reactants).difference(partial_reaction.products)
            coming = set(partial_reaction.products).difference(partial_reaction.reactants)

            bare_reaction_center = set(partial_reaction.compose().center_atoms)

            rule_reac = [x.substructure(set(reaction_center).intersection(x),
                                        as_query=True) for x in reactants if set(reaction_center).intersection(x)]
            rule_prod = [x.substructure(set(reaction_center).intersection(x),
                                        as_query=True) for x in products if set(reaction_center).intersection(x)]

            rule = ReactionContainer(reactants=rule_reac,
                                     products=rule_prod,
                                     meta=reaction.meta)

            for molecule in rule.molecules():
                molecule._rings_sizes = {x: () for x in
                                         molecule._rings_sizes}  # getting rid of the ring sizes info (NEC.)
                molecule._hydrogens = {x: () for x in molecule._hydrogens}  # getting rid of the hydrogen info (NEC.)

            for at_env in set(reaction_center).difference(cleavage.union(coming).union(bare_reaction_center)):

                for x in rule.reactants:
                    if at_env in x.atoms_numbers:
                        x._neighbors[at_env] = ()

                for x in rule.products:
                    if at_env in x.atoms_numbers:
                        x._neighbors[at_env] = ()

            for del_hyb_atom in cleavage:
                for molecule in rule.reactants:
                    if del_hyb_atom in molecule:
                        molecule._hybridization[
                            del_hyb_atom] = ()  # getting rid of the hybridization info in react. (NEC.)

            for del_hyb_atom in coming:
                for molecule in rule.products:
                    if del_hyb_atom in molecule:
                        molecule._hybridization[
                            del_hyb_atom] = ()  # getting rid of the hybridization info in prod. (NEC.)

            rule.flush_cache()
            rules.append(rule)

    for rule in rules:
        if not CC_append_filter(rule):  # checking for one dynamic bond (C-C bond formation)
            rule.meta.update({"maybebad": {"type": ["CC_append_filter"]}})
        if not CC_break_filter(rule):  # checking for one dynamic link (break C-C link)
            if "maybebad" in rule.meta:
                rule.meta["maybebad"]["type"].append("CC_break_filter")
            else:
                rule.meta.update({"maybebad": {"type": ["CC_break_filter"]}})
        if not num_atoms_filter(rule):  # sum of atoms in rule (reactant atoms + product atoms) > 2
            if "maybebad" in rule.meta:
                rule.meta["maybebad"]["type"].append("num_atoms_filter")
            else:
                rule.meta.update({"maybebad": {"type": ["num_atoms_filter"]}})
        if not num_dynbonds_filter(rule):  # number of dynamic links in the rule > 1 and < 8
            if "maybebad" in rule.meta:
                rule.meta["maybebad"]["type"].append("num_dynbonds_filter")
            else:
                rule.meta.update({"maybebad": {"type": ["num_dynbonds_filter"]}})
        if not prod_to_reac_filter(rule,
                                   reaction):  # check. the applic. of the rule on the prod. in order to obt. the react.
            if "maybebad" in rule.meta:
                rule.meta["maybebad"]["type"].append("prod_to_reac_filter")
            else:
                rule.meta.update({"maybebad": {"type": ["prod_to_reac_filter"]}})
        if not reac_to_prod_filter(rule,
                                   reaction):  # check. the applic. of the rule on the react. in order to obt. the prod.
            if "maybebad" in rule.meta:
                rule.meta["maybebad"]["type"].append("reac_to_prod_filter")
            else:
                rule.meta.update({"maybebad": {"type": ["reac_to_prod_filter"]}})
    if save_bad:
        return rules
    else:
        return [rule for rule in rules if "maybebad" not in rule.meta]
