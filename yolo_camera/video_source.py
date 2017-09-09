import os

class VideoSource:
  """ base class for video sources
  """
  def get_launch(self):
    """ get launch string fragment for this video source
    """
    pass

class TestSource(VideoSource):
  def get_launch(self):
    return "videotestsrc"

class V4l2Source:
  def __init__(self,devicename):
    self.devicename = devicename
  def get_launch(self):
    return "v4l2src device=%s" % self.devicename

class PictureSource:
  def __init__(self,picturepath):
    self.picturepath = picturepath
  def get_launch(self):
    return 'multifilesrc location="%s" index=0 caps="image/jpeg,framerate=(fraction)12/1" ! jpegdec ' % self.picturepath
