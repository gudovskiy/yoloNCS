#! /usr/bin/env python
"""
Ncs is a class wrapper for the Movidius Neural Compute device (a USB stick)
"""
import logging
from datetime import datetime
from mvnc import mvncapi as mvnc

logger = logging.getLogger(__name__)


class Ncs(object):
    """

    """

    def __init__(self, device_id=None, graph_path=None):
        # if no device_id is provided, default to the first available NCS device.
        if device_id is None:
            devices = Ncs.enumerate_devices()
            if len(devices) == 0:
                raise RuntimeError('No Movidius NCS devices were found.')
            else:
                self._device_id = devices[0]
                logger.info('Defaulting to Movidius NCS device_id={}'.format(self._device_id))
        else:
            self._device_id = device_id
        self._device = None
        self._graph = None
        self._graph_path = graph_path

    def __enter__(self):
        logger.info('Opening Movidius NCS device={}'.format(self._device_id))
        self._device = mvnc.Device(self._device_id)
        self._device.OpenDevice()
        self.load_graph(graph_path=self._graph_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._graph is not None:
            logger.info('Deallocating graph')
            self._graph.DeallocateGraph()
            self._graph = None
        logger.info('Releasing Movidius NCS device={}'.format(self._device_id))
        self._device.CloseDevice()

    @classmethod
    def enumerate_devices(cls):
        """
        Obtain a list of all Movidius NCS USB devices that are installed on this host.
        :return: string list of devices
        """
        return mvnc.EnumerateDevices()

    @classmethod
    def set_log_level(cls, level):
        mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, level)  # 0=nothing, 1=errors, 2=verbose

    def load_graph(self, graph_path=None, iterations=1, nonblocking=False):
        """
        Loads a pre-compiled NCS graph file into the device
        :param graph_path: Filename of graph to load.
        :param iterations: Count of iterations that should be done on the graph.
        :param nonblocking: If False, calls to LoadTensor and GetResult will block.
        :return: None
        """
        logger.info('Loading Graph {} ...'.format(self._graph_path))
        load_start = datetime.now()
        with open(self._graph_path, mode='rb') as f:
            self._graph = self._device.AllocateGraph(f.read())
        logger.info('Graph options: iteration_count={} nonblocking={}'.format(iterations, nonblocking))
        self._graph.SetGraphOption(mvnc.GraphOption.ITERATIONS, iterations)
        self._graph.SetGraphOption(mvnc.GraphOption.DONT_BLOCK, nonblocking)
        load_duration = datetime.now() - load_start
        logger.info('Graph loading completed after {:.3f} sec.'.format(load_duration.total_seconds()))

    def infer(self, tensor, user_ctx='default_ctx'):
        # Perform the inference on NCS hardware.
        # Input tensor data buffer which contains 16bit half-precision floats (per IEEE 754 half
        # precision binary floating-point format: binary16). The values in the buffer are dependent on the
        # CNN (graph).
        start = datetime.now()
        self._graph.LoadTensor(tensor, user_ctx)
        out, userobj = self._graph.GetResult()
        end = datetime.now()
        inference_time = self._graph.GetGraphOption(mvnc.GraphOption.TIME_TAKEN)
        logger.info(
            'NCS reports inference time={:.3f} ms over {} layers'.format(sum(inference_time), len(inference_time)))
        elapsed_time = end - start
        logger.info('Soft inference time is {:.3f} ms'.format(elapsed_time.total_seconds() * 1000))
        return out, userobj

