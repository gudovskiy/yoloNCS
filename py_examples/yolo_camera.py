#!/usr/bin/env python3
import argparse
import sys
import os
import mvnc.mvncapi as fx

import gi
gi.require_version('Gst', '1.0')
gi.require_version('Gdk', '3.0')
gi.require_version('Gtk', '3.0')
gi.require_version('GLib','2.0')
gi.require_version('GstVideo', '1.0')

from gi.repository import Gdk
from gi.repository import GdkX11
from gi.repository import Gst
from gi.repository import Gtk
from gi.repository import GstVideo
from gi.repository import GLib

import signal

from yolo_camera.video_source import *
from yolo_camera.ncs_network import *
from yolo_camera.ncs_video_sink import *
from yolo_camera.gtk_output import *
from yolo_camera.ncs_thread_model import *

def window_closed (widget, event, pipeline):
  # print("window closed")
  widget.hide()
  pipeline.set_state(Gst.State.NULL)
  Gtk.main_quit ()

def sigint_handler (signal, frame):
  Gtk.main_quit ()

if __name__=="__main__":
  Gdk.init([])
  Gtk.init([])
  Gst.init([])

  fx.SetGlobalOption(fx.GlobalOption.LOGLEVEL, 0)

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  sourcegroup = parser.add_mutually_exclusive_group()

  # VideoSource subclasses as choices for source
  sourcegroup.add_argument('--v4l2-src',
                           help="v4l2 source device name, e.g. /dev/video0",
                           default="/dev/video0")
  sourcegroup.add_argument('--src',
                      choices=[cls.__name__ for cls in vars()['VideoSource'].__subclasses__()],
                      help='Video source')
  sourcegroup.add_argument('--picture-src',
                           help="filename of input picture, e.g. ../../images/dog.jpg")

  # directory for NcsNetwork
  parser.add_argument('-g','--graphdir',
                      help='directory containing the graph, categories.txt, inputsize.txt and stat.txt',
                      default="../networks/YoloTiny")


  # Ncs devices as choices for destination
  ncs_names = fx.EnumerateDevices()
  parser.add_argument('-d','--dest',
                      help='Name of the NCS device to use for inference',
                      choices=ncs_names,
                      default=ncs_names[0] if ncs_names else "default")
  parser.add_argument('--log-level',
                      help="API logging level",
                      type=int)
  parser.add_argument('-v','--verbose',
                      help="Print out additional information",
                      action='store_true')
  parser.add_argument('--opengl',
                      help="use OpenGL instead of Xv extension for preview",
                      action='store_true')
  args = parser.parse_args()

  if args.log_level:
    fx.SetGlobalOption(fx.GlobalOption.LOGLEVEL, args.log_level)

  verbose = args.verbose

  # construct objects representing the parts of the pipeline

  if args.src is not None:
    source = vars()[args.src]()
  elif args.picture_src is not None:
    source = PictureSource(args.picture_src)
  elif args.v4l2_src is not None:
    source = V4l2Source(args.v4l2_src)
  else:
    print("No video source selected")
    sys.exit(1)

  # sink type selection
  sinktype = "xv"    # XFree86 video output plugin using Xv extension
  if args.opengl:
    sinktype = "gl"  # A videosink based on OpenGL

  network = NcsNetwork(args.graphdir)
  viewsink = GtkPreviewSink("view",sinktype)
  appsink = NcsAppsink(network,"app")

  # build Gstreamer launch string

  source2tee = "%s ! tee name=t" % source.get_launch()
  tee2view   = "t. ! %s" % viewsink.get_launch()
  tee2app    = "t. ! %s" % appsink.get_launch()
  launch     = "%s %s %s" % (source2tee, tee2view, tee2app)

  if verbose:
    print(launch)
  try:
    pipeline = Gst.parse_launch( launch )
  except Exception as e:
    print("Could not build video pipeline. Is the camera plugged in and does the display driver support %s ?" % ("OpenGL" if args.opengl else "Xv extension") )
    sys.exit(1)

  # update objects with the created pipeline

  viewsink.set_pipeline(pipeline)
  appsink.set_pipeline(pipeline)

  # build GUI

  window = Gtk.Window()
  window.connect("delete-event", window_closed, pipeline)
  window.set_default_size (448, 448)
  window.set_title (os.path.basename(os.path.normpath(args.graphdir)))

  box = Gtk.Box()
  box.set_spacing(5)
  box.set_orientation(Gtk.Orientation.VERTICAL)
  window.add(box)

  box.pack_start(viewsink, True, True, 0)
  output = OutputWidget()
  box.pack_start(output, False, True, 0)

  #rect = Gdk.Rectangle(0,0,32,32)
  #gtk.gdk.Rectangle(0,0,32,32)
  # Initialize Ncs device

  if verbose:
    print("Opening device")

  dev = fx.Device(args.dest)
  try:
    dev.OpenDevice()
  except Exception as e:
    print("Failed to open NCS. Is the device plugged in?")
    sys.exit(1)

  if verbose:
    print("Loading graph")

  graph = dev.AllocateGraph(network.get_graph_binary())

  # Open UI after device initialization

  if verbose:
    print("Opening UI")

  window.show_all()
  window.realize()
  viewsink.realize()

  # Initialize input and output threads for Ncs

  worker = GlibNcsWorker(fx, graph, appsink, output.put_output)
  worker.start()

  if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
      print("Could not open video pipeline. Is the camera plugged in and does the display driver support " "OpenGL" if args.opengl else "Xv extension" )
  else:
      Gst.debug_bin_to_dot_file (pipeline,Gst.DebugGraphDetails.ALL,'playing-pipeline')    # export GST_DEBUG_DUMP_DOT_DIR=/tmp/
      signal.signal(signal.SIGINT, sigint_handler)
      Gtk.main()
      Gst.debug_bin_to_dot_file (pipeline,Gst.DebugGraphDetails.ALL,'shutting-down-pipeline')

  pipeline.set_state(Gst.State.NULL)
  if verbose:
    print("exiting main loop")
  graph.DeallocateGraph()
  dev.CloseDevice()
  if verbose:
    print("ncs closed")
  worker.stop()
