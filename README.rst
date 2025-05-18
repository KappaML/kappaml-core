.. image:: https://readthedocs.org/projects/kappaml-core/badge/?version=latest
    :alt: ReadTheDocs
    :target: https://kappaml-core.readthedocs.io/en/stable/
.. image:: https://img.shields.io/pypi/v/kappaml-core.svg
    :alt: PyPI-Server
    :target: https://pypi.org/project/kappaml-core/
.. image:: https://img.shields.io/conda/vn/conda-forge/kappaml-core.svg
    :alt: Conda-Forge
    :target: https://anaconda.org/conda-forge/kappaml-core
.. .. image:: https://coveralls.io/repos/github/KappaML/kappaml-core/badge.svg?branch=master
..     :alt: Coveralls
..     :target: https://coveralls.io/github/KappaML/kappaml-core?branch=master
.. image:: https://pepy.tech/badge/kappaml-core/month
    :alt: Monthly Downloads
    :target: https://pepy.tech/project/kappaml-core

============
kappaml-core
============


    Core library for the KappaML project.


This library implements experimental online automated machine learning algorithms for the KappaML project.


Examples
========

MetaClassifier
-------------

.. code-block:: python

    from river.tree import HoeffdingTreeClassifier
    from river.linear_model import LogisticRegression
    from kappaml_core.meta import MetaClassifier

    # Create base models
    models = [
        HoeffdingTreeClassifier(weighted=True),
        HoeffdingTreeClassifier(weighted=False),
        LogisticRegression()
    ]

    # Initialize meta-classifier
    model = MetaClassifier(
        models=models,
        meta_learner=HoeffdingTreeClassifier(),
        metric=Accuracy(),
        mfe_groups=["general"],
        window_size=200,
        meta_update_frequency=50
    )

    for x, y in stream:
        # Make prediction
        y_pred = model.predict_one(x)

        # Update the model
        model.learn_one(x, y)

MetaRegressor
------------

.. code-block:: python

    from river.linear_model import LinearRegression
    from river.tree import HoeffdingTreeRegressor
    from kappaml_core.meta import MetaRegressor

    # Create base models
    models = [
        LinearRegression(),
        StandardScaler() | LinearRegression(),
        [LinearRegression(l2=l2) for l2 in range(0, 1, 0.1)]
        HoeffdingTreeRegressor()
    ]

    # Initialize meta-regressor
    model = MetaRegressor(
        models=models,
        meta_learner=HoeffdingTreeRegressor(),
        metric=MAPE(),
        mfe_groups=["general"],
        window_size=200,
        meta_update_frequency=50
    )

    for x, y in stream:
        # Make prediction
        y_pred = model.predict_one(x)

        # Update the model
        model.learn_one(x, y)


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.1.4. For details and usage
information on PyScaffold see https://pyscaffold.org/.
