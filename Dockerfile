from gsiupm/senpy:0.8.7-python3.5
# from nuig/senpy:0.7.0-dev3-python3.5

COPY logo-Insight.png /usr/local/lib/python3.5/site-packages/senpy/static/img/gsi.png
RUN perl -i -pe s^http://www.gsi.dit.upm.es^https://nuig.insight-centre.org/unlp/^g /usr/local/lib/python3.5/site-packages/senpy/templates/index.html
RUN perl -i -pe 's^https://nuig.insight-centre.org/unlp/" target="_blank"><img id="mixedemotions-logo^http://mixedemotions-project.eu/" target="_blank"><img id="mixedemotions-logo^g' /usr/local/lib/python3.5/site-packages/senpy/templates/index.html
