"""The forwarder that runs inside the sandbox (the network proxy and installer modes) uses only names its imports
define: a rename once turned the stdlib's ``socket.SOL_SOCKET`` into a name the socket module does not have, and the
forwarder failed at its first line."""

import ast
import socket

from unify import sandbox


def test_every_socket_name_the_forwarder_uses_exists():
    tree = ast.parse(sandbox._FORWARDER_SRC)
    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "socket"
    }
    assert "SOL_SOCKET" in used
    assert sorted(n for n in used if not hasattr(socket, n)) == []


def test_the_forwarder_source_compiles():
    compile(sandbox._FORWARDER_SRC, "<forwarder>", "exec")
