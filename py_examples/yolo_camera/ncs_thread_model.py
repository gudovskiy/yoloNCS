from queue import Queue
from gi.repository import GLib
from threading import Thread

class GlibNcsWorker:
  """ Ncs thread model implementation
  """
  def __init__(self, fx, graph, appsink, callback):
      self.running = True
      self.graph = graph
      self.updateq = Queue()
      self.callback = callback
      self.appsink = appsink
      self.fx = fx

  def input_thread(self):
    """ input thread function
    for getting frames from the video
    and loading them into Ncs
    """
    frame_number = 0
    while self.running:
      nb = self.appsink.get_sample()
      if nb is not None: # TODO: eliminate busy looping before samples are available
        try:
          self.graph.LoadTensor(nb,"frame %s" % frame_number)
          frame_number=frame_number + 1
        except Exception as e:
          print("LoadTensor",e)
          pass
    # print("input done")

  def output_thread(self):
    """ output thread function
    for getting inference results from Ncs
    running graph specific post processing of inference result
    queuing the results for main thread callbacks
    """
    while self.running:
      try:
        out, cookie = self.graph.GetResult()
        self.updateq.put((self.appsink.postprocess(out), cookie))
        GLib.idle_add(self.update_ui)
      except Exception as e:
        print("GetResult",e)
        pass
    # print("output done")

  def update_ui(self):
    """ Dispatch callbacks with post processed inference results
    in the main thread context
    """
    while not self.updateq.empty():
      (out, cookie) = self.updateq.get()
      self.callback(cookie, out)
    return False

  def start(self):
    """ start threads and idle handler for callback dispatching
    """
    self.it = Thread(target = self.input_thread)
    self.it.start()
    self.ot = Thread(target = self.output_thread)
    self.ot.start()

  def stop(self):
    """ stop threads
    """
    self.running = False;
    self.it.join()
    self.ot.join()
