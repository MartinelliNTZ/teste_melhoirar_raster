<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>

<qgis autoRefreshTime="0" version="3.34.12-Prizren" maxScale="0" hasScaleBasedVisibilityFlag="0" minScale="1e+08" autoRefreshMode="Disabled" styleCategories="AllStyleCategories">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
    <Private>0</Private>
  </flags>
  <temporal fetchMode="0" enabled="0" mode="0">
    <fixedRange>
      <start/>
      <end/>
    </fixedRange>
  </temporal>
  <elevation zoffset="0" symbology="Line" enabled="0" band="1" zscale="1">
    <data-defined-properties>
      <Option type="Map">
        <Option type="QString" name="name" value=""/>
        <name type="QString" name="name" value=""/>
        <type type="QString" name="type" value="collection"/>
      </Option>
    </data-defined-properties>
  </elevation>
  <customproperties>
    <Option type="Map">
      <Option type="bool" name="WMSBackgroundLayer" value="false"/>
      <Option type="bool" name="WMSPublishDataSourceUrl" value="false"/>
      <Option type="int" name="embeddedWidgets/count" value="0"/>
      <Option type="QString" name="identify/format" value="Value"/>
    </Option>
  </customproperties>
  <mapTip enabled="1"/>
  <pipe-data-defined-properties>
    <Option type="Map">
      <Option type="QString" name="name" value=""/>
      <properties/>
      <type type="QString" name="type" value="collection"/>
    </Option>
  </pipe-data-defined-properties>
  <pipe>
    <provider>
      <resampling zoomedOutResamplingMethod="nearestNeighbour" enabled="false" maxOversampling="2" zoomedInResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer blueBand="3" alphaBand="-1" type="multibandcolor" redBand="1" nodataColor="" greenBand="2" opacity="1.0">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <redContrastEnhancement>
        <minValue>0.0154631</minValue>
        <maxValue>0.1502903</maxValue>
        <algorithm>StretchToMinimumMaximum</algorithm>
      </redContrastEnhancement>
      <greenContrastEnhancement>
        <minValue>0.0154631</minValue>
        <maxValue>0.1502903</maxValue>
        <algorithm>StretchToMinimumMaximum</algorithm>
      </greenContrastEnhancement>
      <blueContrastEnhancement>
        <minValue>0.0154631</minValue>
        <maxValue>0.1502903</maxValue>
        <algorithm>StretchToMinimumMaximum</algorithm>
      </blueContrastEnhancement>
    </rasterrenderer>
    <brightnesscontrast brightness="0" gamma="1" contrast="0"/>
    <huesaturation colorizeGreen="128" grayscaleMode="0" invertColors="0" colorizeStrength="100" colorizeBlue="128" colorizeRed="255" saturation="0" colorizeOn="0"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>