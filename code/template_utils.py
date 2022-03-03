"""Template filters"""
from CGRtools.containers import ReactionContainer, QueryContainer, MoleculeContainer
from CGRtools.reactor import CGRReactor
from CGRtools.exceptions import InvalidAromaticRing


def CC_append_filter(rule: ReactionContainer) -> bool:
    """
    Checking for the presence of only one dynamic bond between carbon atoms (i.e.,
    such rules are removed where the formation of a C-C bond is carried out). Most
    often this is necessary because of the occurrence in the atom-atom mapping.
    :param rule: input reaction transformation rule
    :return: bool
    """
    reactantq = QueryContainer()
    for reactant in rule.reactants:
        reactantq = reactantq.union(reactant)
    productq = QueryContainer()
    for product in rule.products:
        productq = productq.union(product)
    bond_break = []
    for n, m, b in reactantq.bonds():
        if productq.has_atom(n) and productq.has_atom(m):
            if not productq.has_bond(n, m):
                bond_break.append((n, m, b))
        elif productq.has_atom(n) or productq.has_atom(m):
            bond_break.append((n, m, b))
    bond_append = []
    bond_append_other = []
    for n, m, b in productq.bonds():
        if productq.atom(n).atomic_symbol == 'C' and productq.atom(m).atomic_symbol == 'C':
            if reactantq.has_atom(n) and reactantq.has_atom(m):
                if not reactantq.has_bond(n, m):
                    bond_append.append((n, m, b))
            elif reactantq.has_atom(n) or reactantq.has_atom(m):
                bond_append.append((n, m, b))
        else:
            if reactantq.has_atom(n) and reactantq.has_atom(m):
                if not reactantq.has_bond(n, m):
                    bond_append_other.append((n, m, b))
            elif reactantq.has_atom(n) or reactantq.has_atom(m):
                bond_append_other.append((n, m, b))
    if len(bond_break) == 0 and len(bond_append) == 1 and len(bond_append_other) == 0:
        return False
    else:
        return True


def CC_break_filter(rule: ReactionContainer) -> bool:
    """
    Checking for the presence of only one dynamic bond between carbon atoms (i.e., such rules are removed where
    the C-C bond is broken). Most often this is necessary because of the occurrence in the atom-atom mapping.
    :param rule: input reaction transformation rule
    :return: bool
    """
    reactantq = QueryContainer()
    for reactant in rule.reactants:
        reactantq = reactantq.union(reactant)
    productq = QueryContainer()
    for product in rule.products:
        productq = productq.union(product)
    bond_break = []
    bond_break_other = []
    for n, m, b in reactantq.bonds():
        if reactantq.atom(n).atomic_symbol == 'C' and reactantq.atom(m).atomic_symbol == 'C':
            if productq.has_atom(n) and productq.has_atom(m):
                if not productq.has_bond(n, m):
                    bond_break.append((n, m, b))
            elif productq.has_atom(n) or productq.has_atom(m):
                bond_break.append((n, m, b))
        else:
            if productq.has_atom(n) and productq.has_atom(m):
                if not productq.has_bond(n, m):
                    bond_break_other.append((n, m, b))
            elif productq.has_atom(n) or productq.has_atom(m):
                bond_break_other.append((n, m, b))
    bond_append = []
    for n, m, b in productq.bonds():
        if reactantq.has_atom(n) and reactantq.has_atom(m):
            if not reactantq.has_bond(n, m):
                bond_append.append((n, m, b))
        elif reactantq.has_atom(n) or reactantq.has_atom(m):
            bond_append.append((n, m, b))
    if len(bond_break) == 1 and len(bond_append) == 0 and len(bond_break_other) == 0:
        return False
    else:
        return True


def prod_to_reac_filter(rule: ReactionContainer,
                        r: ReactionContainer) -> bool:
    """
    Checking the application of the transformation rule to the product.
    The correctness of the generated rule is checked.
    :param rule: input reaction transformation rule
    :param r: input reaction
    :return: bool
    """
    q = QueryContainer()
    for prod in rule.products:
        q = q.union(prod)

    template = ReactionContainer(reactants=[q], products=rule.reactants)
    reactor = CGRReactor(template)
    m_product = MoleculeContainer()
    for prod in r.products:
        m_product = m_product.union(prod)
    m_product.canonicalize()
    flag_total = False

    for proded in reactor(m_product, automorphism_filter=False):
        try:
            proded.canonicalize()
            proded.flush_cache()
        except InvalidAromaticRing:
            continue
        else:
            list_proded = proded.split()
            new_react = []
            try:
                for react in r.reactants:
                    for re in react.split():
                        re.canonicalize()
                        re.flush_cache()
                        new_react.append(re)

                    for prod_res in list_proded:
                        prod_res.canonicalize()
                        prod_res.flush_cache()
                        if any(prod_res == reactant for reactant in new_react):
                            return True
            except InvalidAromaticRing:
                continue
    else:
        if flag_total is False:
            return False


def reac_to_prod_filter(rule: ReactionContainer,
                        r: ReactionContainer) -> bool:
    """
    Checking the application of the transformation rule to the reactants.
    The correctness of the generated rule is checked.
    :param rule: input reaction transformation rule
    :param r: input reaction
    :return: bool
    """
    q = QueryContainer()
    for react in rule.reactants:
        q = q.union(react)

    template = ReactionContainer(reactants=[q], products=rule.products)
    reactor = CGRReactor(template)
    m_reactant = MoleculeContainer()
    for react in r.reactants:
        m_reactant = m_reactant.union(react)
    m_reactant.canonicalize()
    flag_total = False

    for proded in reactor(m_reactant, automorphism_filter=False):
        try:
            proded.canonicalize()
            proded.flush_cache()
        except InvalidAromaticRing:
            continue
        else:
            list_proded = proded.split()
            new_react = []
            try:
                for react in r.products:
                    for re in react.split():
                        re.canonicalize()
                        re.flush_cache()
                        new_react.append(re)

                    for prod_res in list_proded:
                        prod_res.canonicalize()
                        prod_res.flush_cache()
                        if any(prod_res == reactant for reactant in new_react):
                            return True
            except InvalidAromaticRing:
                continue
    else:
        if flag_total is False:
            return False


def num_atoms_filter(rule: ReactionContainer) -> bool:
    """
    The rules are tested for a small number of atoms in the rule transformation reaction. It turned out that
    most of the diatomic rules are halogen rules, which are useless, so these rules need to be removed.
    :param rule: input reaction transformation rule
    :return: bool
    """
    num_atoms_rule = sum([x.atoms_count for x in rule.reactants]) + sum([x.atoms_count for x in rule.products])
    if num_atoms_rule < 3:
        return False
    else:
        return True


def num_dynbonds_filter(rule: ReactionContainer) -> bool:
    """
    Check for a large number of dynamic bonds (more than 7). The problem may lie in the incorrect atom-to-atom mapping.
    Also, it is checked that the number of dynamic links is not equal to 1.
    :param rule: input reaction transformation rule
    :return: bool
    """
    new_reactants = []
    for reactant in rule.reactants:
        r1 = MoleculeContainer()
        for n, a in reactant.atoms():
            r1.add_atom(a.atomic_number, _map=n)
        for n, m, b in reactant.bonds():
            r1.add_bond(n, m, int(b))
        new_reactants.append(r1)
    new_products = []
    for product in rule.products:
        r1 = MoleculeContainer()
        for n, a in product.atoms():
            r1.add_atom(a.atomic_number, _map=n)
        for n, m, b in product.bonds():
            r1.add_bond(n, m, int(b))
        new_products.append(r1)
    new_reaction = ReactionContainer(reactants=new_reactants, products=new_products)
    cgr = ~new_reaction
    num_dyn_bonds = len(cgr.center_bonds)
    if num_dyn_bonds == 1 or num_dyn_bonds > 7:
        return False
    else:
        return True
