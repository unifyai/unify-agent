"""The memory lanes: the local relay that gives the capture fork, the writer and the writer's helpers their own model
route when no external one is set.

* :mod:`.relay_server`: the relay process. It holds the provider key in its own memory only, serves one route per lane
  on loopback, admits a call only with that lane's token and for that lane's models, meters every call into the lane's
  journal and holds all lanes to one spend cap. Standard library only; it is run by path, never imported by the
  harness.
* :mod:`.relay`: starts and stops that process (:class:`.relay.LaneRelay`).
* :mod:`.launch`: the ``unify`` command's side. With memory on and no lane given by the environment, the command
  starts the relay, then runs the controller as its child with each lane's base URL and token on an inherited
  descriptor, and stops the writer service and the relay when the controller ends.

Nothing is imported here, so importing one module never loads the others.
"""
