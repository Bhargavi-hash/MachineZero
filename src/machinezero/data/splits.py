from __future__ import annotations

from dataclasses import dataclass

from machinezero.aliencpu.generator import generate_architecture


@dataclass(frozen=True)
class ArchitectureSplits:
    train: tuple[int,...]; validation: tuple[int,...]; test: tuple[int,...]

def make_splits(base_seed:int=1000,n_train:int=64,n_validation:int=16,n_test:int=16)->ArchitectureSplits:
    seeds=tuple(base_seed+i for i in range(n_train+n_validation+n_test))
    return ArchitectureSplits(seeds[:n_train],seeds[n_train:n_train+n_validation],seeds[n_train+n_validation:])

def architecture_ids(seeds:tuple[int,...])->set[str]: return {generate_architecture(s).architecture_id for s in seeds}

def validate_disjoint(s:ArchitectureSplits)->None:
    a,b,c=architecture_ids(s.train),architecture_ids(s.validation),architecture_ids(s.test)
    if a&b or a&c or b&c: raise ValueError('Architecture-level split overlap')
