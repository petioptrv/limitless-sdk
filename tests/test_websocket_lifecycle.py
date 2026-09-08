"""Tests for WebSocket connection-lifecycle handling."""

import pytest

from limitless_sdk.websocket import WebSocketClient
from limitless_sdk.websocket.client import DEFAULT_NAMESPACE
from limitless_sdk.websocket.types import WebSocketConfig, WebSocketState


class _FakeAsyncClient:
    """Fake python-socketio client that records handlers per namespace."""

    def __init__(self, *args, **kwargs):
        self.handlers = {}
        self.emits = []
        self.shutdown_called = False

    def on(self, event, handler=None, namespace=None):
        if handler is not None:
            self.handlers[(namespace, event)] = handler
            return handler

        def decorator(func):
            self.handlers[(namespace, event)] = func
            return func

        return decorator

    async def connect(self, url, headers=None, transports=None, namespaces=None, wait_timeout=None):
        handler = self.handlers.get((DEFAULT_NAMESPACE, "connect"))
        if handler:
            await handler()

    async def emit(self, channel, options=None, namespace=None):
        self.emits.append({"channel": channel, "options": options, "namespace": namespace})

    async def disconnect(self):
        pass

    async def shutdown(self):
        self.shutdown_called = True

    async def fire(self, event, *args):
        """Invoke a handler the way python-socketio dispatches it."""
        handler = self.handlers.get((DEFAULT_NAMESPACE, event))
        assert handler is not None, f"no handler registered on {DEFAULT_NAMESPACE} for {event!r}"
        return await handler(*args)


@pytest.fixture
def fake_sio(monkeypatch):
    fake = _FakeAsyncClient()
    monkeypatch.setattr(
        "limitless_sdk.websocket.client.AsyncClient",
        lambda *args, **kwargs: fake,
    )
    return fake


@pytest.mark.asyncio
async def test_internal_handlers_attach_to_markets_namespace(fake_sio):
    client = WebSocketClient(WebSocketConfig(auto_reconnect=False))

    await client.connect()

    for event in ("connect", "disconnect", "connect_error"):
        assert (DEFAULT_NAMESPACE, event) in fake_sio.handlers
        assert (None, event) not in fake_sio.handlers


@pytest.mark.asyncio
async def test_drop_marks_reconnecting_and_reconnect_restores_subscriptions(fake_sio):
    client = WebSocketClient(WebSocketConfig(auto_reconnect=True))
    await client.connect()
    await client.subscribe("subscribe_market_prices", {"marketSlugs": ["market-a"]})
    fake_sio.emits.clear()

    await fake_sio.fire("disconnect", "transport error")
    assert client.state == WebSocketState.RECONNECTING
    assert client.is_connected() is False

    await fake_sio.fire("connect")
    assert client.state == WebSocketState.CONNECTED
    assert client._reconnect_attempts == 1
    assert fake_sio.emits == [
        {
            "channel": "subscribe_market_prices",
            "options": {"marketSlugs": ["market-a"]},
            "namespace": DEFAULT_NAMESPACE,
        }
    ]


@pytest.mark.asyncio
async def test_reconnect_restores_every_distinct_subscription(fake_sio):
    client = WebSocketClient(WebSocketConfig(auto_reconnect=True))
    await client.connect()
    await client.subscribe("subscribe_market_prices", {"marketSlugs": ["market-a"]})
    await client.subscribe("subscribe_market_prices", {"marketSlugs": ["market-b"]})
    assert len(client._subscriptions) == 2
    fake_sio.emits.clear()

    await fake_sio.fire("disconnect", "transport error")
    await fake_sio.fire("connect")

    resubscribed = [e["options"]["marketSlugs"] for e in fake_sio.emits]
    assert sorted(resubscribed) == [["market-a"], ["market-b"]]


@pytest.mark.asyncio
async def test_intentional_disconnect_is_not_reported_as_a_drop(fake_sio):
    client = WebSocketClient(WebSocketConfig(auto_reconnect=True))
    await client.connect()

    await client.disconnect()

    assert fake_sio.shutdown_called is True
    assert client.state == WebSocketState.DISCONNECTED
    assert client.is_connected() is False
    assert client._subscriptions == {}


@pytest.mark.asyncio
async def test_disconnect_can_preserve_subscriptions_for_credential_rotation(fake_sio):
    client = WebSocketClient(WebSocketConfig(auto_reconnect=True))
    await client.connect()
    await client.subscribe("subscribe_market_prices", {"marketSlugs": ["market-a"]})

    await client.disconnect(preserve_subscriptions=True)

    assert list(client._subscriptions) == ["subscribe_market_prices:market-a"]


@pytest.mark.asyncio
async def test_auto_reconnect_disabled_reports_disconnected_on_drop(fake_sio):
    client = WebSocketClient(WebSocketConfig(auto_reconnect=False))
    await client.connect()

    await fake_sio.fire("disconnect", "transport error")

    assert client.state == WebSocketState.DISCONNECTED


def test_subscription_keys_do_not_collide_across_markets():
    client = WebSocketClient(WebSocketConfig(auto_reconnect=False))
    key = client._get_subscription_key

    assert key("subscribe_market_prices", {"marketSlugs": ["a"]}) != key(
        "subscribe_market_prices", {"marketSlugs": ["b"]}
    )
    # Order-insensitive so the same set of markets maps to one key.
    assert key("subscribe_market_prices", {"marketSlugs": ["a", "b"]}) == key(
        "subscribe_market_prices", {"marketSlugs": ["b", "a"]}
    )
    # Legacy singular form and the address variants still key correctly.
    assert key("subscribe_market_prices", {"marketSlug": "a"}).endswith(":a")
    assert key("subscribe_positions", {}).endswith(":global")
    assert key("subscribe_market_prices", {"marketAddresses": ["0xabc"]}).endswith(":0xabc")
