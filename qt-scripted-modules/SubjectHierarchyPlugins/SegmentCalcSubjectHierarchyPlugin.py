import vtk, qt, ctk, slicer, os, sys
import logging
from AbstractScriptedSubjectHierarchyPlugin import *

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMPORTDIR = BASE_DIR+'\\SegmentCalcDir\\'
sys.path.append(IMPORTDIR)

class SegmentCalcSubjectHierarchyPlugin(AbstractScriptedSubjectHierarchyPlugin):
  """ Scripted subject hierarchy plugin for the Segment Editor module.

      This is also an example for scripted plugins, so includes all possible methods.
      The methods that are not needed (i.e. the default implementation in
      qSlicerSubjectHierarchyAbstractPlugin is satisfactory) can simply be
      omitted in plugins created based on this one.
  """

  # Necessary static member to be able to set python source to scripted subject hierarchy plugin
  filePath = __file__

  def __init__(self, scriptedPlugin):
    scriptedPlugin.name = 'SegmentCalc'
    AbstractScriptedSubjectHierarchyPlugin.__init__(self, scriptedPlugin)
    path = BASE_DIR +'\\SegmentCalcDir\\model'

    files = os.listdir(path)
    pthlist=list()
    self.segmentCalcActions=list()
    for f in files:
      if f.endswith('.pth'):
        contextstr = "segCMR by " + f
        segmentCalcAction = qt.QAction(contextstr, scriptedPlugin)
        segmentCalcAction.connect("triggered()", lambda: self.onSegment(f))
        self.segmentCalcActions.append(segmentCalcAction)

  def canAddNodeToSubjectHierarchy(self, node, parentItemID):
    # This plugin cannot own any items (it's not a role but a function plugin),
    # but the it can be decided the following way:
    # if node is not None and node.IsA("vtkMRMLMyNode"):
    #   return 1.0
    return 0.0

  def canOwnSubjectHierarchyItem(self, itemID):
    # This plugin cannot own any items (it's not a role but a function plugin),
    # but the it can be decided the following way:
    # pluginHandlerSingleton = slicer.qSlicerSubjectHierarchyPluginHandler.instance()
    # shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    # associatedNode = shNode.GetItemDataNode(itemID)
    # if associatedNode is not None and associatedNode.IsA("vtkMRMLMyNode"):
    #   return 1.0
    return 0.0

  def roleForPlugin(self):
    # As this plugin cannot own any items, it doesn't have a role either
    return "N/A"

  def helpText(self):
    # return ("<p style=\" margin-top:4px; margin-bottom:1px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
      # "<span style=\" font-family:'sans-serif'; font-size:9pt; font-weight:600; color:#000000;\">"
      # "SegmentEditor module subject hierarchy help text"
      # "</span>"
      # "</p>"
      # "<p style=\" margin-top:0px; margin-bottom:11px; margin-left:26px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
      # "<span style=\" font-family:'sans-serif'; font-size:9pt; color:#000000;\">"
      # "This is how you can add help text to the subject hierarchy module help box via a python scripted plugin."
      # "</span>"
      # "</p>\n")
    return ""

  def icon(self, itemID):
    # As this plugin cannot own any items, it doesn't have an icon either
    # import os
    # iconPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons/MyIcon.png')
    # if self.canOwnSubjectHierarchyItem(itemID) > 0.0 and os.path.exists(iconPath):
      # return qt.QIcon(iconPath)
    # Item unknown by plugin
    return qt.QIcon()

  def itemContextMenuActions(self):
    return self.segmentCalcActions

  def onSegment(self, modelname):
    pluginHandlerSingleton = slicer.qSlicerSubjectHierarchyPluginHandler.instance()
    currentItemID = pluginHandlerSingleton.currentItem()

    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    volumeNode = shNode.GetItemDataNode(currentItemID)

    inputfile = BASE_DIR+'\\SegmentCalcDir\\tmpdata\\in.nrrd'
    outputfile = BASE_DIR+'\\SegmentCalcDir\\tmpdata\\out.nrrd'

    slicer.util.saveNode(volumeNode, inputfile)
    from SegmentCalcDir.test import runMain
    runMain(inputfile, outputfile, modelname)

    predict = slicer.util.loadSegmentation(outputfile)
    os.remove(inputfile)
    os.remove(outputfile)
    
    # Switch to Segment Editor module

    # Create new segmentation only if there is no segmentation node, or the current segmentation is not empty
    # (switching to the module will create an empty segmentation if there is none in the scene, but not otherwise)
    # segmentationNode = editorWidget.parameterSetNode.GetSegmentationNode()
    segmentationNode = predict
    # Name segmentation node based on the volume
    segmentationNode.SetName(volumeNode.GetName() + '_CalculatedSeg')

    # Set master volume

    # Place segmentation under the master volume in subject hierarchy
    segmentationShItemID = shNode.GetItemByDataNode(segmentationNode)
    shNode.SetItemParent(segmentationShItemID, shNode.GetItemParent(currentItemID))

  def sceneContextMenuActions(self):
    return []

  def showContextMenuActionsForItem(self, itemID):
    # Scene
    if not itemID:
      # No scene context menu actions in this plugin
      return

    # Volume but not LabelMap
    pluginHandlerSingleton = slicer.qSlicerSubjectHierarchyPluginHandler.instance()
    if ( pluginHandlerSingleton.pluginByName('Volumes').canOwnSubjectHierarchyItem(itemID)
         and not pluginHandlerSingleton.pluginByName('LabelMaps').canOwnSubjectHierarchyItem(itemID) ):
      # Get current item
      currentItemID = pluginHandlerSingleton.currentItem()
      if not currentItemID:
        logging.error("Invalid current item")
        return
      for segAction in self.segmentCalcActions:
        segAction.visible = True

  def tooltip(self, itemID):
    # As this plugin cannot own any items, it doesn't provide tooltip either
    return ""

  def setDisplayVisibility(self, itemID, visible):
    pluginHandlerSingleton = slicer.qSlicerSubjectHierarchyPluginHandler.instance()
    pluginHandlerSingleton.pluginByName('Default').setDisplayVisibility(itemID, visible)

  def getDisplayVisibility(self, itemID):
    pluginHandlerSingleton = slicer.qSlicerSubjectHierarchyPluginHandler.instance()
    return pluginHandlerSingleton.pluginByName('Default').getDisplayVisibility(itemID)
