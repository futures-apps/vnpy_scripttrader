from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Event, Thread
from typing import cast
import unittest
from unittest.mock import patch

from vnpy.event import EventEngine
from vnpy.trader.constant import Direction, Exchange, Product
from vnpy.trader.engine import MainEngine
from vnpy.trader.object import ContractData, OrderRequest

from vnpy_scripttrader.engine import ScriptEngine


class FakeLogEngine:
    def register_log(self, event_type: str) -> None:
        self.event_type: str = event_type


class FakeMainEngine:
    def __init__(self, contract: ContractData | None) -> None:
        self.contract: ContractData | None = contract
        self.log_engine: FakeLogEngine = FakeLogEngine()
        self.order_requests: list[OrderRequest] = []
        self.gateway_names: list[str] = []
        self.started: Event = Event()
        self.release: Event = Event()

    def get_engine(self, engine_name: str) -> FakeLogEngine | None:
        if engine_name == "log":
            return self.log_engine
        return None

    def get_contract(self, vt_symbol: str) -> ContractData | None:
        if self.contract is not None and vt_symbol == self.contract.vt_symbol:
            return self.contract
        return None

    def send_order(self, req: OrderRequest, gateway_name: str) -> str:
        self.order_requests.append(req)
        self.gateway_names.append(gateway_name)
        return f"{gateway_name}.1"


class FakeEventEngine:
    def __init__(self) -> None:
        self.events: list = []

    def put(self, event: object) -> None:
        self.events.append(event)


def make_contract() -> ContractData:
    return ContractData(
        symbol="rb2510",
        exchange=Exchange.SHFE,
        name="rb2510",
        product=Product.FUTURES,
        size=10,
        pricetick=1,
        gateway_name="CTP",
    )


class TestBuyOrder(unittest.TestCase):
    def test_buy_sends_order_request(self) -> None:
        contract: ContractData = make_contract()
        main_engine: FakeMainEngine = FakeMainEngine(contract)
        engine: ScriptEngine = ScriptEngine(
            cast(MainEngine, main_engine),
            cast(EventEngine, FakeEventEngine()),
        )
        price: float = 3500.0
        volume: float = 2.0

        orderid: str = engine.buy(contract.vt_symbol, price, volume)

        assert orderid == "CTP.1"
        assert main_engine.gateway_names == [contract.gateway_name]
        assert len(main_engine.order_requests) == 1
        req: OrderRequest = main_engine.order_requests[0]
        assert req.symbol == contract.symbol
        assert req.direction == Direction.LONG
        assert req.price == price
        assert req.volume == volume

    def test_buy_without_contract_does_not_send(self) -> None:
        main_engine: FakeMainEngine = FakeMainEngine(None)
        engine: ScriptEngine = ScriptEngine(
            cast(MainEngine, main_engine),
            cast(EventEngine, FakeEventEngine()),
        )

        orderid: str = engine.buy("rb2510.SHFE", 3500.0, 2.0)

        assert orderid == ""
        assert main_engine.order_requests == []

    def test_run_strategy_resets_active_flag_after_script_returns(self) -> None:
        main_engine: FakeMainEngine = FakeMainEngine(None)
        engine: ScriptEngine = ScriptEngine(
            cast(MainEngine, main_engine),
            cast(EventEngine, FakeEventEngine()),
        )
        with TemporaryDirectory() as directory:
            script_path: Path = Path(directory) / "script_reset_flag_test.py"
            script_path.write_text("def run(engine):\n    pass\n", encoding="utf-8")
            engine.strategy_active = True

            engine.run_strategy(str(script_path))

        assert engine.strategy_active is False


class ScriptLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.main = FakeMainEngine(None)
        self.events = FakeEventEngine()
        self.engine = ScriptEngine(cast(MainEngine, self.main), cast(EventEngine, self.events))
        self.addCleanup(self.stop_engine)

    def stop_engine(self) -> None:
        self.main.release.set()
        self.engine.stop_strategy(timeout=2)

    def script(self, source: str, name: str = "sample.py") -> Path:
        path = self.directory / name
        path.write_text(source, encoding="utf-8")
        return path

    def finish(self) -> None:
        thread = self.engine.strategy_thread
        self.assertIsNotNone(thread)
        if thread:
            thread.join(2)
            self.assertFalse(thread.is_alive())

    def test_top_level_runs_once_and_source_changes_are_reloaded(self) -> None:
        output = self.directory / "result.txt"
        source = (
            "from pathlib import Path\n"
            "path = Path(__file__).with_name('result.txt')\n"
            "with path.open('a') as f: f.write('A')\n"
            "def run(engine): pass\n"
        )
        script = self.script(source)
        self.engine.start_strategy(str(script))
        self.finish()
        self.assertEqual("A", output.read_text())
        self.assertEqual("completed", self.engine.strategy_status)
        self.assertFalse(self.engine.strategy_active)
        script.write_text(source.replace("'A'", "'B'"), encoding="utf-8")
        self.engine.start_strategy(str(script))
        self.finish()
        self.assertEqual("AB", output.read_text())

    def test_same_basename_uses_selected_absolute_path(self) -> None:
        other = self.directory / "other"
        other.mkdir()
        first = self.script("def run(engine): engine.write_log('first')\n")
        second = other / first.name
        second.write_text("def run(engine): engine.write_log('second')\n", encoding="utf-8")
        for path in (first, second):
            self.engine.start_strategy(str(path))
            self.finish()
        messages = [event.data.msg for event in self.events.events]
        self.assertEqual(1, messages.count("first"))
        self.assertEqual(1, messages.count("second"))

    def test_exception_is_reported_and_next_script_can_start(self) -> None:
        bad = self.script("def run(engine): raise ValueError('test failure')\n")
        self.engine.start_strategy(str(bad))
        self.finish()
        self.assertEqual("failed", self.engine.strategy_status)
        self.assertIn("ValueError: test failure", self.engine.strategy_error)
        good = self.script("def run(engine): pass\n", "good.py")
        self.engine.start_strategy(str(good))
        self.finish()
        self.assertEqual("completed", self.engine.strategy_status)
        self.assertEqual("", self.engine.strategy_error)

    def test_duplicate_start_and_cooperative_stop(self) -> None:
        path = self.script(
            "from time import sleep\n"
            "def run(engine):\n"
            "    engine.main_engine.started.set()\n"
            "    while engine.strategy_active: sleep(0.001)\n"
        )
        self.engine.start_strategy(str(path))
        self.assertTrue(self.main.started.wait(2))
        with self.assertRaises(RuntimeError):
            self.engine.start_strategy(str(path))
        self.engine.stop_strategy(timeout=2)
        self.assertFalse(self.engine.strategy_active)
        self.assertIsNone(self.engine.strategy_thread)
        self.assertEqual("stopped", self.engine.strategy_status)

    def test_stop_timeout_preserves_thread_and_blocks_restart(self) -> None:
        path = self.script(
            "def run(engine):\n"
            "    engine.main_engine.started.set()\n"
            "    engine.main_engine.release.wait(5)\n"
        )
        self.engine.start_strategy(str(path))
        self.assertTrue(self.main.started.wait(2))
        with self.assertRaises(TimeoutError):
            self.engine.stop_strategy(timeout=0.01)
        self.assertEqual("stopping", self.engine.strategy_status)
        self.assertFalse(self.engine.strategy_active)
        with self.assertRaises(RuntimeError):
            self.engine.start_strategy(str(path))
        self.main.release.set()
        self.finish()
        self.assertEqual("stopped", self.engine.strategy_status)
        self.engine.stop_strategy()
        self.assertIsNone(self.engine.strategy_thread)

    def test_script_can_stop_itself_without_joining_its_own_thread(self) -> None:
        path = self.script("def run(engine): engine.stop_strategy()\n")
        self.engine.start_strategy(str(path))
        self.finish()
        self.assertEqual("stopped", self.engine.strategy_status)

    def test_start_thread_failure_resets_running_state(self) -> None:
        path = self.script("def run(engine): pass\n")
        with patch.object(Thread, "start", side_effect=RuntimeError("thread unavailable")):
            with self.assertRaises(RuntimeError):
                self.engine.start_strategy(str(path))
        self.assertFalse(self.engine.strategy_active)
        self.assertIsNone(self.engine.strategy_thread)
        self.assertEqual("failed", self.engine.strategy_status)

    def test_close_stops_cooperative_script(self) -> None:
        path = self.script(
            "from time import sleep\n"
            "def run(engine):\n"
            "    engine.main_engine.started.set()\n"
            "    while engine.strategy_active: sleep(0.001)\n"
        )
        self.engine.start_strategy(str(path))
        self.assertTrue(self.main.started.wait(2))
        self.engine.close()
        self.assertIsNone(self.engine.strategy_thread)

    def test_concurrent_start_runs_only_one_script(self) -> None:
        path = self.script(
            "def run(engine):\n"
            "    engine.main_engine.started.set()\n"
            "    engine.main_engine.release.wait(5)\n"
        )
        barrier = Barrier(3)
        outcomes = []

        def start() -> None:
            barrier.wait(timeout=2)
            try:
                self.engine.start_strategy(str(path))
                outcomes.append("started")
            except RuntimeError:
                outcomes.append("rejected")

        callers = [Thread(target=start) for _ in range(2)]
        for caller in callers:
            caller.start()
        barrier.wait(timeout=2)
        for caller in callers:
            caller.join(2)
            self.assertFalse(caller.is_alive())
        self.assertCountEqual(["started", "rejected"], outcomes)
        self.assertTrue(self.main.started.wait(2))

    def test_invalid_entry_point_and_system_exit_are_failures(self) -> None:
        for source in (
            "run = 1",
            "def run(): pass",
            "async def run(engine): pass",
            "def run(engine): raise SystemExit(1)",
        ):
            with self.subTest(source=source):
                path = self.script(source)
                self.engine.start_strategy(str(path))
                self.finish()
                self.assertEqual("failed", self.engine.strategy_status)
                self.assertTrue(self.engine.strategy_error)
