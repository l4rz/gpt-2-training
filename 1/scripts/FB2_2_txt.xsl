<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:fb="http://www.gribuser.ru/xml/fictionbook/2.0">
	<xsl:strip-space elements="*"/>
	<xsl:output method="text" encoding="UTF-8"/>
	<xsl:key name="note-link" match="fb:section|fb:p" use="@id"/>

	<xsl:template match="*">
<!-- <xsl:variable/> -->
				<!-- BUILD BOOK -->
<xsl:for-each select="fb:body[not(@name)]">
<xsl:if test="position()!=1">
<xsl:text ></xsl:text>
</xsl:if>

<xsl:if test="@name">
<xsl:value-of select="@name"/>
<xsl:text ></xsl:text>
</xsl:if>
<!-- <xsl:apply-templates /> -->
<xsl:apply-templates/>
</xsl:for-each>
<xsl:text ></xsl:text>
</xsl:template>
<!-- body -->
<xsl:template match="fb:body">
<xsl:text ></xsl:text>
<xsl:apply-templates/>
</xsl:template>

<xsl:template match="fb:section">
<xsl:apply-templates select="./*"/>
</xsl:template>
	
<!-- p -->
<xsl:template match="fb:p">
<xsl:apply-templates/>
<xsl:text >&#010;</xsl:text>
</xsl:template>

<xsl:template match="fb:p" mode="note">
<xsl:apply-templates/>
</xsl:template>

<xsl:template match="fb:strong|fb:emphasis|fb:style"><xsl:apply-templates/></xsl:template>

<xsl:template match="fb:a">
<xsl:choose>
<xsl:when test="(@type) = 'note'">
<xsl:choose>
<xsl:when test="starts-with(@xlink:href,'#')"><xsl:for-each select="key('note-link',substring-after(@xlink:href,'#'))">[<xsl:apply-templates mode="note"/>]</xsl:for-each></xsl:when>
<xsl:otherwise><xsl:for-each select="key('note-link',@xlink:href)">[<xsl:apply-templates mode="note"/>]</xsl:for-each></xsl:otherwise>
</xsl:choose>
</xsl:when>
<xsl:otherwise>
<xsl:apply-templates/>
</xsl:otherwise>
</xsl:choose>
</xsl:template>

<xsl:template match="fb:empty-line">
<xsl:text ></xsl:text>
</xsl:template>

<xsl:template match="fb:image">
<xsl:text ></xsl:text>
</xsl:template>

<xsl:template match="fb:poem">
<xsl:apply-templates/>
</xsl:template>

	<!-- stanza -->
<xsl:template match="fb:stanza">
<xsl:apply-templates/>
<xsl:text >&#010;</xsl:text>
</xsl:template>
	<!-- v -->
<xsl:template match="fb:v">
<xsl:text >		</xsl:text>
<xsl:apply-templates/>
<xsl:text >&#010;</xsl:text>
</xsl:template>

</xsl:stylesheet>
