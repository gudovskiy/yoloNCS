from gi.repository import Gtk

class GtkPreviewSink(Gtk.DrawingArea):
  """ GTK DrawingArea with Gstreamer sink
  """
  def __init__(self,name,sinktype='xv'):
    """ name - name of the gstreamer element
    """
    super().__init__()
    self.set_double_buffered(True)
    self.sinktype=sinktype
    self.name = name

  def get_launch(self):
    """ get Gstreamer launch string fragment for the sink
    """
    return "\
    queue max-size-buffers=2 leaky=downstream ! \
    %simagesink name=%s" % (self.sinktype ,self.name)

  def set_pipeline(self,pipeline):
    """ set associated Gstreamer pipeline
    """
    self.pipeline = pipeline

  def realize(self):      # TODO: implement proper widget subclassing or 'prepare-window-handle' signal handler
    self.pipeline.get_by_name(self.name).set_window_handle(self.get_window().get_xid())

class OutputWidget(Gtk.Label):
  """ a GtkLabel which displays results from Ncs Inference
  """
  def __init__(self):
    super().__init__()
    self.set_justify(Gtk.Justification.LEFT)
    self.set_text("-")

  def put_output(self, userobj, out):
    """ Method for receiving the (postprocessed) results
    userobj - user object passed to the NcsExpress
    out - output
    """
    text = ( "<tt>" + userobj + "\n" + "\n".join([ "%80s" % cat[:79] +
                                                   ' (' + '{0:.2f}'.format(probability*100) + '%) '
                                                   for (cat,probability) in out.top ]) + "</tt>")
    self.set_markup(text)

