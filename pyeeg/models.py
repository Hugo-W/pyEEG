# -*- coding: utf-8 -*-
"""
Desctription
~~~~~~~~~~~~

In this module, we can find different method to model the relationship
between stimulus and (EEG) response. Namely there are wrapper functions
implementing:

    - Forward modelling (stimulus -> EEG), a.k.a _TRF_ (Temporal Response Functions)
    - Backward modelling (EEG -> stimulus)
    - CCA
    - ?

TODO
''''
Maybe add DNN models, if so this should rather be a subpackage.
Modules for each modelling architecture will then be implemented within the subpackage
and we would have in `__init__.py` an entry to load all architectures.


"""
