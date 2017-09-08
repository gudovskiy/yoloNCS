from gi.repository import Gst

class NcsAppsink:
  """ Gstreamer appsink for streaming video frames into a Ncs graph
  """
  def __init__(self,network,name):
    """ network - Ncs graph object
    name - gstreamer name attribute for the appsink object. just needs to be unique
    """
    self.network = network
    self.name = name

  def set_pipeline(self,pipeline):
    """ set associated Gstreamer pipeline
    """
    self.pipeline = pipeline
    self.appsink = self.pipeline.get_by_name(self.name)

  def get_launch(self):
    """ get Gstreamer launch string fragment for appsink
    """
    (width, height, format ) = self.network.rawinputformat()
    return "\
    videoscale ! video/x-raw, width=%s, height=%s ! \
    videoconvert ! video/x-raw, format=%s ! \
    appsink name=%s max-buffers=1 drop=true enable-last-sample=true" % (width, height, format, self.name )

  def get_sample(self):
    """ get a preprocessed frame to be pushed to the graph
    """
    sample = self.appsink.get_property('last-sample')
    if sample:
        buf = sample.get_buffer()
        res, info = buf.map(Gst.MapFlags.READ)
        nb = self.network.preprocess(info.data)
        buf.unmap(info)
        del buf
        del sample
        return nb
    return None

  def postprocess(self,out):
    """ graph specific postprocessing of the inference result
    """
    return self.network.postprocess(out)
