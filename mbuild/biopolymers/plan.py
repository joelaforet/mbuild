"""What a modification will do, before it does it: ``ModificationPlan``.

``Protein.plan_attach``, ``plan_mutate``, ``plan_protonate`` and
``plan_deprotonate`` read their arguments into a plan and change
nothing. The plan names the atoms that leave, the bonds that form,
break or change order, the charges that change and the hydrogens that
are created, in the words of the ``Compound`` edits that
``Protein.apply`` then makes. ``str(plan)`` is the plan as text and
``plan.draw()`` is the same chemistry drawn with RDKit: the site
residue, the fragment and the product, with the leaving atoms and the
new bonds marked. ``apply`` makes the edits, relaxes the structure as
the plan's options say, and returns what the one-call verb returns.

This is the two-step form of the verbs. ``attach(...)`` is
``apply(plan_attach(...))``, and the same for the other three.
"""

from dataclasses import dataclass, field

from mbuild.exceptions import MBuildError

__all__ = ["ModificationPlan"]

_ORDER_NAMES = {1.0: "1", 2.0: "2", 3.0: "3", 1.5: "aromatic"}


@dataclass(eq=False)
class ModificationPlan:
    """The edits one modification makes, as particles of the compounds.

    Attributes
    ----------
    kind : str
        ``"attach"``, ``"mutate"``, ``"protonate"`` or ``"deprotonate"``.
    protein : Protein
        The protein the plan was made for; ``apply`` refuses another.
    site : Residue
        The residue that is modified.
    site_atom : mbuild.Compound or None
        The protein atom that forms the bond (``CA`` for a mutation,
        the named atom for a protonation).
    fragment : mbuild.Compound or None
        A clone of the fragment, made when the plan was. ``apply``
        places this clone, so the caller's fragment is never changed.
    fragment_atom : mbuild.Compound or None
        The fragment atom that bonds to ``site_atom``.
    leaving : list of (mbuild.Compound, mbuild.Compound)
        ``(kept atom, leaving atom)`` pairs. The leaving atom and every
        atom beyond it, away from the kept atom, are removed.
    groups : dict of mbuild.Compound -> tuple of str
        For each leaving atom, the names of the whole group that goes
        with it, the leaving atom first.
    formed : list of (mbuild.Compound, mbuild.Compound, float)
        Bonds to add, with their order.
    removed : list of (mbuild.Compound, mbuild.Compound)
        Bonds to break between atoms that both stay.
    reordered : list of (mbuild.Compound, mbuild.Compound, float)
        Bonds that stay and take a new order.
    charges : dict of mbuild.Compound -> int
        Formal charges the modification sets.
    new_hydrogens : list of mbuild.Compound
        Atoms that gain a hydrogen the modification creates.
    reaction : str or None
        The reaction SMARTS an ``attach`` follows, resolved from its
        name; None for the one-bond rule and for the other kinds.
    notes : list of str
        Facts about the plan that the fields above do not say, such as
        a hydrogen that changes partners or the name a residue takes.
    options : dict
        The geometry and bookkeeping options the verb was called with
        (``relax``, ``minimize``, ``merge``, ``platform``), which
        ``apply`` follows after the edits.
    applied : bool
        Whether ``apply`` has run this plan. A plan runs once.
    """

    kind: str = "attach"
    protein: object = None
    site: object = None
    site_atom: object = None
    fragment: object = None
    fragment_atom: object = None
    leaving: list = field(default_factory=list)
    groups: dict = field(default_factory=dict)
    formed: list = field(default_factory=list)
    removed: list = field(default_factory=list)
    reordered: list = field(default_factory=list)
    charges: dict = field(default_factory=dict)
    new_hydrogens: list = field(default_factory=list)
    reaction: str = None
    notes: list = field(default_factory=list)
    options: dict = field(default_factory=dict)
    applied: bool = False
    _state: dict = field(default_factory=dict, repr=False)

    def cross_bonds(self, side_particles):
        """Return the formed bonds with one atom in ``side_particles``."""
        side = set(side_particles)
        return [bond for bond in self.formed if (bond[0] in side) != (bond[1] in side)]

    def _add_leaving(self, kept, leaving, group, append=True):
        """Record a leaving atom with the whole group that goes with it."""
        if append:
            self.leaving.append((kept, leaving))
        # The leaving atom first, then the rest by name, so the text is
        # the same in every process whatever order the graph walk took.
        names = sorted(particle.name for particle in group if particle is not leaving)
        self.groups[leaving] = (leaving.name, *names)
        self._state.setdefault("gone", {})[leaving] = list(group)

    # -- text --------------------------------------------------------

    def _in_fragment(self, atom):
        # Read from the set made when the plan was, not from atom.root:
        # after apply the fragment's atoms belong to the protein.
        return atom in self._state.get("in_fragment", ())

    def _label(self, atom):
        """Name an atom the way errors do: ``LYS A:48 NZ`` or ``OC8 C1``."""
        from mbuild.biopolymers.residue import _pdb_label

        residue = atom.parent
        if self._in_fragment(atom):
            several = len(self._fragment_residues()) > 1
            head = f"{residue.name} {residue.resnum}" if several else residue.name
            return f"{head} {atom.name}"
        return f"{_pdb_label(residue)} {atom.name}"

    def _fragment_residues(self):
        from mbuild.biopolymers.residue import Residue

        if self.fragment is None:
            return []
        if isinstance(self.fragment, Residue):
            return [self.fragment]
        return [c for c in self.fragment.successors() if isinstance(c, Residue)] or [
            self.fragment
        ]

    def _group_text(self, atom):
        names = self.groups.get(atom, (atom.name,))
        if len(names) == 1:
            return names[0]
        rest = names[1:]
        if len(rest) > 6:
            return f"{names[0]} (+{len(rest)} atoms)"
        return f"{names[0]} (+{' '.join(rest)})"

    def _bond_text(self, atom1, atom2, order=None):
        text = f"{atom1.name}-{atom2.name}"
        if order is not None:
            text += f" ({_ORDER_NAMES.get(float(order), f'{order:g}')})"
        return text

    def _header(self):
        """Return the one-line heading of the plan."""
        from mbuild.biopolymers.residue import _pdb_label

        site = _pdb_label(self.site) if self.site is not None else "?"
        if self.kind == "attach":
            frag = self._label(self.fragment_atom) if self.fragment_atom else "?"
            head = f"attach: {site} {self.site_atom.name} + {frag}"
            if self.reaction is not None:
                name = self._state.get("reaction_name")
                how = f'reaction "{name}"' if name else "reaction"
                head = f"attach by {how}: {site} {self.site_atom.name} + {frag}"
            return head
        if self.kind == "mutate":
            return f"mutate: {site} -> {self._state.get('new_name', '?')}"
        atom = self.site_atom.name if self.site_atom is not None else "?"
        return f"{self.kind}: {site} {atom}"

    def summary(self):
        """Return the plan as one line, for a log."""
        parts = [self._header()]
        if self.leaving:
            parts.append(
                "leaves " + ", ".join(self._group_text(a) for _, a in self.leaving)
            )
        if self.formed:
            parts.append(
                "forms "
                + ", ".join(self._bond_text(a, b, o) for a, b, o in self.formed)
            )
        if self.new_hydrogens:
            parts.append(
                "adds H on " + ", ".join(atom.name for atom in self.new_hydrogens)
            )
        if self._state.get("noop"):
            parts.append("nothing to do")
        return "; ".join(parts)

    def __str__(self):
        lines = [self._header()]

        def row(key, text):
            lines.append(f"  {key:<9} {text}")

        if self.reaction is not None:
            row("reaction", self.reaction)
        if self._state.get("noop"):
            row("nothing", self._state["noop"])
            return "\n".join(lines)
        protein_side = [a for _, a in self.leaving if not self._in_fragment(a)]
        frag_side = [a for _, a in self.leaving if self._in_fragment(a)]
        if protein_side or frag_side:
            text = []
            if protein_side:
                text.append(
                    "protein: " + ", ".join(self._group_text(a) for a in protein_side)
                )
            if frag_side:
                text.append(
                    "fragment: " + ", ".join(self._group_text(a) for a in frag_side)
                )
            row("leaves", " | ".join(text))
        if self.formed:
            row("forms", ", ".join(self._bond_text(a, b, o) for a, b, o in self.formed))
        if self.removed:
            row("breaks", ", ".join(self._bond_text(a, b) for a, b in self.removed))
        if self.reordered:
            row(
                "reorders",
                ", ".join(
                    f"{self._bond_text(a, b)} -> {_ORDER_NAMES.get(float(o), f'{o:g}')}"
                    for a, b, o in self.reordered
                ),
            )
        if self.charges:
            row(
                "charges",
                ", ".join(
                    f"{atom.name}: {self._charge_now(atom):+d} -> {charge:+d}"
                    for atom, charge in self.charges.items()
                ),
            )
        if self.new_hydrogens:
            row("adds H", "on " + ", ".join(atom.name for atom in self.new_hydrogens))
        for note in self.notes:
            row("note", note)
        then = self._then()
        if then:
            row("then", then)
        return "\n".join(lines)

    def _charge_now(self, atom):
        return getattr(atom.parent, "atom_formal_charges", {}).get(atom.name, 0)

    def _then(self):
        """Describe what ``apply`` does after the edits."""
        if self.kind in ("protonate", "deprotonate"):
            return ""
        opts = self.options
        steps = []
        if self.kind == "attach":
            steps.append("align the fragment on ports")
            if len(self._state.get("cross", ())) > 1:
                steps.append("rigid fit to every formed bond")
        elif self._state.get("closure") is not None:
            steps.append("superpose the component on the backbone")
        elif self.fragment is None:
            steps.append("place the hydrogen")
        else:
            steps.append("align the side chain on ports")
        if not opts.get("relax", True):
            steps.append("keep the rigid placement (relax=False)")
        elif not opts.get("minimize", True):
            steps.append("turn it clear of clashes; minimize later (minimize=False)")
        else:
            steps.append("turn it clear of clashes and minimize the neighbourhood")
        if self.kind == "attach":
            if opts.get("merge"):
                steps.append("merge the atoms into the site residue (no bond record)")
            else:
                steps.append("record the bond")
        return "; ".join(steps)

    # -- freshness ---------------------------------------------------

    def _check_fresh(self, protein):
        """Raise unless ``apply`` can still make these edits on ``protein``."""
        if self.applied:
            raise MBuildError(
                "This plan has already been applied. Plan the modification "
                "again to apply it a second time."
            )
        if self.protein is not protein:
            raise MBuildError(
                "The plan was made for another Protein. Plan the modification "
                "on this one."
            )
        stale = (
            "The protein changed since this plan was made (an atom it names "
            "is gone or no longer bonded as planned). Plan the modification "
            "again."
        )
        atoms = [self.site_atom] if self.site_atom is not None else []
        atoms += [kept for kept, _ in self.leaving]
        atoms += [
            a for bond in self.formed + self.removed + self.reordered for a in bond[:2]
        ]
        atoms += list(self.charges) + list(self.new_hydrogens)
        for atom in atoms:
            root = self.fragment.root if self._in_fragment(atom) else protein
            if atom.root is not root or not root.bond_graph.has_node(atom):
                raise MBuildError(stale)
        for kept, leaving in self.leaving:
            root = self.fragment.root if self._in_fragment(leaving) else protein
            if not root.bond_graph.has_edge(kept, leaving):
                raise MBuildError(stale)
        for atom1, atom2 in [b[:2] for b in self.removed + self.reordered]:
            root = self.fragment.root if self._in_fragment(atom1) else protein
            if not root.bond_graph.has_edge(atom1, atom2):
                raise MBuildError(stale)
        if self.site is not None and self.site.root is not protein:
            raise MBuildError(stale)
        # A protonation plan reads the residue's template variant; a
        # plan to do nothing is also stale once that variant changed.
        if (
            "template" in self._state
            and self.site.template is not self._state["template"]
        ):
            raise MBuildError(stale)

    # -- drawing -----------------------------------------------------

    def _leaving_atoms(self):
        """Every atom that leaves, as a set."""
        gone = set()
        for group in self._state.get("gone", {}).values():
            gone.update(group)
        return gone

    def _panel(self, compound, particles, mark_red, mark_green, hydrogens):
        """Build one drawing panel: an RDKit mol, its labels and colours."""
        from mbuild.biopolymers.residue import _rdkit_mol

        interesting = set(mark_red) | set(mark_green)
        for bond in self.formed + self.removed + self.reordered:
            interesting.update(bond[:2])
        interesting.update(self.new_hydrogens)
        keep = []
        for particle in particles:
            if hydrogens or particle.element.symbol != "H" or particle in interesting:
                keep.append(particle)
                continue
            # A hydrogen on an atom that reacts stays visible, so the
            # drawing shows what the reacting atom carries.
            if any(n in interesting for n in particle.direct_bonds()):
                keep.append(particle)
        editable, index = _rdkit_mol(compound, keep)
        # _rdkit_mol gives every atom no implicit hydrogens, so a heavy
        # atom whose hydrogens are hidden is told how many it carries,
        # or sanitization would draw radicals on it.
        present, kept = set(particles), set(keep)
        for particle in keep:
            if particle.element.symbol == "H":
                continue
            hidden = sum(
                1
                for n in particle.direct_bonds()
                if n.element.symbol == "H" and n in present and n not in kept
            )
            if hidden:
                editable.GetAtomWithIdx(index[particle]).SetNumExplicitHs(hidden)
        colors = {}
        for particle in keep:
            if particle in mark_red:
                colors[index[particle]] = (1.0, 0.35, 0.35)
            elif particle in mark_green:
                colors[index[particle]] = (0.45, 0.85, 0.45)
        mol = editable.GetMol()
        for particle, i in index.items():
            mol.GetAtomWithIdx(i).SetProp("atomNote", particle.name)
        return mol, index, colors

    def _product(self, hydrogens):
        """Build the product panel by applying the edits to RDKit copies."""
        from rdkit import Chem

        gone = self._leaving_atoms()
        site_particles = [p for p in self.site.particles() if p not in gone]
        frag_particles = [
            p
            for residue in self._fragment_residues()
            for p in residue.particles()
            if p not in gone
        ]
        new_bonds = set()
        for atom1, atom2, _ in self.formed:
            new_bonds.update((atom1, atom2))
        green = new_bonds | set(self.new_hydrogens)
        site_mol, site_index, colors = self._panel(
            self.protein, site_particles, set(), green, hydrogens
        )
        index = dict(site_index)
        mol = site_mol
        if frag_particles:
            frag_mol, frag_index, frag_colors = self._panel(
                self.fragment.root, frag_particles, set(), green, hydrogens
            )
            offset = mol.GetNumAtoms()
            mol = Chem.CombineMols(mol, frag_mol)
            index.update({p: i + offset for p, i in frag_index.items()})
            colors.update({i + offset: c for i, c in frag_colors.items()})
        editable = Chem.RWMol(mol)
        for atom1, atom2 in self.removed:
            if atom1 in index and atom2 in index:
                editable.RemoveBond(index[atom1], index[atom2])
        for atom1, atom2, order in self.formed:
            if atom1 in index and atom2 in index:
                editable.AddBond(index[atom1], index[atom2], _rdkit_order(order))
        for atom1, atom2, order in self.reordered:
            if atom1 in index and atom2 in index:
                bond = editable.GetBondBetweenAtoms(index[atom1], index[atom2])
                if bond is not None:
                    bond.SetBondType(_rdkit_order(order))
        for atom, charge in self.charges.items():
            if atom in index:
                editable.GetAtomWithIdx(index[atom]).SetFormalCharge(charge)
        for heavy in self.new_hydrogens:
            if heavy in index:
                new = editable.AddAtom(Chem.Atom(1))
                editable.GetAtomWithIdx(new).SetProp("atomNote", "H")
                editable.GetAtomWithIdx(new).SetNoImplicit(True)
                editable.AddBond(index[heavy], new, Chem.BondType.SINGLE)
                colors[new] = (0.45, 0.85, 0.45)
        return editable.GetMol(), colors

    def draw(self, size=(1000, 340), hydrogens=False):
        """Draw the plan: the site residue, the fragment and the product.

        Atoms that leave are red and atoms that form a bond are green.
        Hydrogens are hidden unless they take part, or sit on an atom
        that does; pass ``hydrogens=True`` to draw them all. In a
        notebook the plan itself displays as this drawing.

        Parameters
        ----------
        size : tuple of int, optional, default=(1000, 340)
            Width and height of the drawing in pixels.
        hydrogens : bool, optional, default=False
            Draw every hydrogen.

        Returns
        -------
        IPython.display.SVG or str
            The drawing, shown inline when returned from a notebook
            cell. The SVG text when IPython is not installed.
        """
        svg = self._render(size, hydrogens, png=False)
        try:
            from IPython.display import SVG
        except ImportError:
            return svg
        return SVG(svg)

    def _repr_svg_(self):
        return self._render((1000, 340), False, png=False)

    def _render(self, size, hydrogens, png):
        from rdkit import Chem
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D

        gone = self._leaving_atoms()
        bonding = set()
        for atom1, atom2, _ in self.formed:
            bonding.update((atom1, atom2))
        bonding.update(self.new_hydrogens)
        if self.site_atom is not None:
            bonding.add(self.site_atom)
        panels = []
        site_mol, _, colors = self._panel(
            self.protein,
            list(self.site.particles()),
            gone,
            bonding,
            hydrogens,
        )
        from mbuild.biopolymers.residue import _pdb_label

        panels.append((site_mol, colors, f"site: {_pdb_label(self.site)}"))
        frag_residues = self._fragment_residues()
        if frag_residues:
            particles = [p for r in frag_residues for p in r.particles()]
            frag_mol, _, colors = self._panel(
                self.fragment.root, particles, gone, bonding, hydrogens
            )
            names = ", ".join(r.name for r in frag_residues[:3])
            panels.append((frag_mol, colors, f"fragment: {names}"))
        if not self._state.get("noop"):
            product, colors = self._product(hydrogens)
            legend = "product"
            if self.kind == "mutate":
                legend = f"product: {self._state.get('new_name', '')}"
            panels.append((product, colors, legend))
        mols = []
        for mol, _, _ in panels:
            try:
                Chem.SanitizeMol(mol)
            except Chem.rdchem.MolSanitizeException:
                mol.UpdatePropertyCache(strict=False)
            rdDepictor.Compute2DCoords(mol)
            mols.append(mol)
        width, height = size
        make = rdMolDraw2D.MolDraw2DCairo if png else rdMolDraw2D.MolDraw2DSVG
        drawer = make(width, height, width // len(panels), height)
        options = drawer.drawOptions()
        options.annotationFontScale = 0.65
        options.legendFontSize = 14
        # The site residue is cut from its chain at N and C; those open
        # valences are not chemistry of the plan, so no radical dots.
        options.includeRadicals = False
        drawer.DrawMolecules(
            mols,
            legends=[legend for _, _, legend in panels],
            highlightAtoms=[list(colors) for _, colors, _ in panels],
            highlightAtomColors=[colors for _, colors, _ in panels],
        )
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    def save(self, filename, size=(1000, 340), hydrogens=False):
        """Write the drawing to ``filename``, as SVG or, for ``.png``, as PNG.

        The PNG needs an RDKit built with Cairo, as the conda-forge
        package is.
        """
        filename = str(filename)
        png = filename.lower().endswith(".png")
        text = self._render(size, hydrogens, png=png)
        with open(filename, "wb" if png else "w") as handle:
            handle.write(text)


def _rdkit_order(order):
    from rdkit import Chem

    return {
        1.0: Chem.BondType.SINGLE,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        1.5: Chem.BondType.AROMATIC,
    }.get(float(order), Chem.BondType.SINGLE)
