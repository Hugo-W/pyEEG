:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

Methods
~~~~~~~

   {% block methods %}

        .. autosummary::
            :toctree:
            :template: method.rst

        {% for item in methods %}
            {%- if not item.startswith('_') or item in ['__call__'] %}
            {{ name }}.{{ item }}
            {%- endif -%}
        {%- endfor %}
   {% endblock %}

.. raw:: html

    <div class="clearer"></div>