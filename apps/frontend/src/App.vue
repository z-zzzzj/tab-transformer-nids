<template>
  <div class="app-shell">
    <header class="topbar">
      <div class="topbar__inner">
        <div class="brand-lockup">
          <p class="eyebrow">TabTransformer NIDS Console</p>
          <div class="brand-copy">
            <h1>网络流量异常检测演示台</h1>
          </div>
        </div>

        <div class="status-cluster">
          <span class="status-pill" :class="{ 'is-online': health.ok }">
            服务 {{ health.ok ? "在线" : "离线" }}
          </span>
          <span class="status-pill" :class="{ 'is-online': wsConnected, 'is-fallback': !wsConnected }">
            {{ transportLabel }}
          </span>
          <span class="status-pill" :class="{ 'is-online': replay.running }">
            {{ replayLabel }}
          </span>
        </div>
      </div>
    </header>

    <main class="dashboard">
      <section class="hero-surface hero-surface--dark">
        <div class="hero-copy">
          <p class="section-kicker">运行概览</p>
          <h2>同屏观察吞吐、告警与当前流细节。</h2>

          <dl class="headline-metrics" :class="{ 'headline-metrics--with-protocol': hasProtocolData }">
            <div>
              <dt>累计事件</dt>
              <dd>{{ formatCount(totalEvents) }}</dd>
            </div>
            <div>
              <dt>攻击占比</dt>
              <dd>{{ formatPercent(attackRatio) }}</dd>
            </div>
            <div v-if="hasProtocolData">
              <dt>主协议</dt>
              <dd>{{ topProtocol.label }}</dd>
            </div>
            <div>
              <dt>主端口</dt>
              <dd>{{ topPort.label }}</dd>
            </div>
          </dl>
        </div>

        <div class="control-surface">
          <div class="control-surface__header">
            <div>
              <p class="section-kicker">回放控制</p>
              <h3>本地演示编排</h3>
            </div>
            <p class="control-status">{{ actionState.text }}</p>
          </div>

          <div class="control-grid">
            <label class="control-field" for="rate">
              <span>目标速度</span>
              <div class="field-input">
                <input id="rate" v-model.number="controls.ratePerSecond" min="1" max="500" type="number" />
                <em>条 / 秒</em>
              </div>
            </label>

            <label class="control-field" for="batch">
              <span>回放批次</span>
              <div class="field-input">
                <input id="batch" v-model.number="controls.batchSize" min="1" max="1000" type="number" />
                <em>条</em>
              </div>
            </label>
          </div>

          <div class="button-row">
            <button class="button button--primary" @click="handleStartReplay">开始回放</button>
            <button class="button button--secondary" @click="handleStopReplay">停止回放</button>
            <button class="button button--ghost" @click="resetDashboard">重置面板</button>
          </div>

          <dl class="control-meta">
            <div>
              <dt>回放状态</dt>
              <dd>{{ replayLabel }}</dd>
            </div>
            <div>
              <dt>数据同步</dt>
              <dd>{{ transportLabel }}</dd>
            </div>
            <div>
              <dt>最近攻击名</dt>
              <dd>{{ latestOriginalLabel }}</dd>
            </div>
          </dl>
        </div>
      </section>

      <section class="main-grid">
        <article class="panel panel--dark throughput-panel">
          <div class="panel-head">
            <div>
              <p class="section-kicker">实时吞吐趋势</p>
              <h3>事件接入速率</h3>
            </div>
            <p>{{ totalEvents === 0 ? "等待事件流" : "最近 60 个时间点" }}</p>
          </div>
          <div ref="throughputRef" class="chart chart--large"></div>
        </article>

        <article class="panel panel--light ratio-panel">
          <div class="panel-head">
            <div>
              <p class="section-kicker">攻击 / 正常比例</p>
              <h3>当前分类结构</h3>
            </div>
            <p>{{ formatCount(state.attackCount) }} 攻击 / {{ formatCount(state.benignCount) }} 正常</p>
          </div>
          <div ref="ratioRef" class="chart"></div>
        </article>

        <article class="panel panel--light distribution-panel">
          <div class="panel-head">
            <div>
              <p class="section-kicker">{{ hasProtocolData ? "协议与端口分布" : "端口分布" }}</p>
              <h3>{{ hasProtocolData ? "高频通信画像" : "高频端口画像" }}</h3>
            </div>
            <p>仅展示当前窗口内的高频项</p>
          </div>
          <div ref="distributionRef" class="chart"></div>
        </article>

        <article class="panel panel--dark alerts-panel">
          <div class="panel-head">
            <div>
              <p class="section-kicker">最近告警列表</p>
              <h3>实时事件队列</h3>
            </div>
            <p>{{ MAX_VISIBLE_ALERTS }} 条</p>
          </div>

          <div class="table-shell">
            <table class="alerts-table">
              <thead>
                <tr>
                  <th>时间</th>
                  <th>结果</th>
                  <th>置信度</th>
                  <th>原始攻击名</th>
                  <th v-if="hasProtocolData">协议</th>
                  <th>端口</th>
                </tr>
              </thead>
              <tbody v-if="recentAlerts.length">
                <tr v-for="(item, index) in recentAlerts" :key="getAlertKey(item, index)">
                  <td>{{ formatTimestamp(item.timestamp) }}</td>
                  <td>
                    <span class="prediction-pill" :class="item.prediction === 'attack' ? 'is-attack' : 'is-benign'">
                      {{ formatPrediction(item.prediction) }}
                    </span>
                  </td>
                  <td>{{ formatPercent(item.confidence) }}</td>
                  <td>{{ item.original_label || "Benign" }}</td>
                  <td v-if="hasProtocolData">{{ normalizeProtocol(item.display_fields?.protocol) }}</td>
                  <td>{{ item.display_fields?.destination_port ?? "-" }}</td>
                </tr>
              </tbody>
              <tbody v-else>
                <tr>
                  <td :colspan="alertTableColumnCount" class="table-empty">尚未收到事件，启动回放后这里会持续刷新。</td>
                </tr>
              </tbody>
            </table>
          </div>
        </article>

        <article class="panel panel--light detail-panel">
          <div class="panel-head">
            <div>
              <p class="section-kicker">当前流详情</p>
              <h3>最新流量样本</h3>
            </div>
            <p>{{ currentFlowTimestamp }}</p>
          </div>

          <div class="detail-layout">
            <div class="detail-facts">
              <div v-for="item in currentFlowFacts" :key="item.label" class="fact-row">
                <span>{{ item.label }}</span>
                <strong>{{ item.value }}</strong>
              </div>
            </div>

            <pre class="detail-json">{{ prettyCurrentFlow }}</pre>
          </div>
        </article>
      </section>
    </main>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from "vue";
import { getHealth, getReplayEvents, getReplayStatus, startReplay, stopReplay } from "./api";
import { initChart, updateChart } from "./composables/useCharts";
import { createAlertSocket } from "./ws";

const CHART_COLORS = Object.freeze({
  blue: "#0071e3",
  blueSoft: "rgba(0, 113, 227, 0.16)",
  blueLine: "rgba(41, 151, 255, 0.9)",
  white: "#f5f5f7",
  whiteMuted: "rgba(245, 245, 247, 0.62)",
  dark: "#1d1d1f",
  darkMuted: "rgba(29, 29, 31, 0.54)",
  darkSoft: "rgba(29, 29, 31, 0.08)",
  track: "#d2d2d7"
});

const throughputRef = ref(null);
const ratioRef = ref(null);
const distributionRef = ref(null);

const health = reactive({ ok: false });
const replay = reactive({ running: false });
const wsConnected = ref(false);

const controls = reactive({
  ratePerSecond: 100,
  batchSize: 100
});

const actionState = reactive({
  text: "等待启动回放。",
  tone: "neutral"
});

const state = reactive({
  throughputTimes: [],
  throughputValues: [],
  benignCount: 0,
  attackCount: 0,
  protocolCounts: {},
  portCounts: {},
  recentAlerts: [],
  currentFlow: null
});

const seenEventKeys = new Set();
const numberFormatter = new Intl.NumberFormat("zh-CN");

let throughputChart = null;
let ratioChart = null;
let distributionChart = null;
let socket = null;
let pollTimer = null;
let statusTimer = null;
let reconnectTimer = null;
let pollCursor = 0;
let isUnmounted = false;

const MAX_VISIBLE_ALERTS = 100;

const recentAlerts = computed(() => state.recentAlerts.slice(0, MAX_VISIBLE_ALERTS));
const totalEvents = computed(() => state.attackCount + state.benignCount);
const attackRatio = computed(() => (totalEvents.value ? state.attackCount / totalEvents.value : 0));
const topProtocol = computed(() => getTopEntry(state.protocolCounts, "未形成主协议"));
const topPort = computed(() => getTopEntry(state.portCounts, "未形成主端口"));
const latestAlert = computed(() => state.recentAlerts[0] ?? null);
const latestOriginalLabel = computed(() => latestAlert.value?.original_label || "暂无");
const hasProtocolData = computed(() => Object.keys(state.protocolCounts).length > 0);
const alertTableColumnCount = computed(() => (hasProtocolData.value ? 6 : 5));
const transportLabel = computed(() => (wsConnected.value ? "实时同步" : "自动同步"));
const replayLabel = computed(() => (replay.running ? "回放进行中" : "回放已停止"));
const currentFlowTimestamp = computed(() =>
  latestAlert.value?.timestamp ? formatTimestamp(latestAlert.value.timestamp) : "等待首条事件"
);
const currentFlowFacts = computed(() => {
  const current = latestAlert.value || {};
  const facts = [
    { label: "预测结果", value: formatPrediction(current.prediction) },
    { label: "置信度", value: formatPercent(current.confidence) },
    { label: "原始攻击名", value: current.original_label || "Benign" },
    { label: "目标端口", value: current.display_fields?.destination_port ?? "-" },
    { label: "阈值", value: current.threshold ?? "-" }
  ];
  const protocol = normalizeProtocol(current.display_fields?.protocol);
  if (protocol) {
    facts.splice(3, 0, { label: "协议", value: protocol });
  }
  return facts;
});
const prettyCurrentFlow = computed(() => {
  if (!state.currentFlow) {
    return "等待回放事件...";
  }
  return JSON.stringify(state.currentFlow, null, 2);
});

function formatCount(value) {
  return numberFormatter.format(Number(value || 0));
}

function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`;
}

function formatPrediction(prediction) {
  return prediction === "attack" ? "攻击" : "正常";
}

function formatTimestamp(value) {
  if (!value) {
    return "--";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return date.toLocaleString("zh-CN", { hour12: false });
}

function getAlertKey(item, index) {
  return item?.event_id || `${item?.timestamp || "event"}-${item?.prediction || "event"}-${index}`;
}

function getTopEntry(record, emptyLabel) {
  const sorted = Object.entries(record).sort((left, right) => right[1] - left[1]);
  if (!sorted.length) {
    return { label: emptyLabel, value: 0 };
  }
  return { label: sorted[0][0] || "-", value: sorted[0][1] };
}

function normalizeProtocol(value) {
  if (value === null || value === undefined) {
    return "";
  }
  const text = String(value).trim();
  if (!text || ["unknown", "UNKNOWN", "nan", "NaN", "null", "None", "-"].includes(text)) {
    return "";
  }
  return text;
}

function setActionMessage(text, tone = "neutral") {
  actionState.text = text;
  actionState.tone = tone;
}

function rememberEvent(event) {
  const key =
    event.event_id ||
    [
      event.timestamp,
      event.prediction,
      event.original_label,
      event.display_fields?.protocol,
      event.display_fields?.destination_port
    ].join("|");

  if (seenEventKeys.has(key)) {
    return false;
  }

  seenEventKeys.add(key);
  if (seenEventKeys.size > 5000) {
    seenEventKeys.clear();
    seenEventKeys.add(key);
  }
  return true;
}

function buildAxisStyle(isDark) {
  return {
    axisLine: { lineStyle: { color: isDark ? "rgba(255,255,255,0.16)" : CHART_COLORS.darkSoft } },
    axisLabel: { color: isDark ? CHART_COLORS.whiteMuted : CHART_COLORS.darkMuted },
    splitLine: { lineStyle: { color: isDark ? "rgba(255,255,255,0.08)" : "rgba(29,29,31,0.08)" } }
  };
}

function getThroughputOption() {
  const axis = buildAxisStyle(true);
  const xAxisLabelInterval = Math.max(0, Math.ceil(state.throughputTimes.length / 6) - 1);

  return {
    animationDuration: 300,
    grid: { left: 48, right: 18, top: 32, bottom: 28 },
    tooltip: {
      trigger: "axis",
      backgroundColor: "rgba(0, 0, 0, 0.82)",
      borderWidth: 0,
      textStyle: { color: CHART_COLORS.white }
    },
    xAxis: {
      type: "category",
      boundaryGap: false,
      data: state.throughputTimes,
      ...axis,
      axisLabel: { ...axis.axisLabel, hideOverlap: true, interval: xAxisLabelInterval }
    },
    yAxis: {
      type: "value",
      name: "events/s",
      nameTextStyle: { color: CHART_COLORS.whiteMuted, padding: [0, 0, 0, 8] },
      ...axis
    },
    series: [
      {
        type: "line",
        smooth: true,
        symbol: "none",
        lineStyle: { width: 3, color: CHART_COLORS.blueLine },
        areaStyle: { color: CHART_COLORS.blueSoft },
        data: state.throughputValues
      }
    ]
  };
}

function getRatioOption() {
  const benignValue = state.benignCount;
  const attackValue = state.attackCount;
  const ratioData = [];

  if (attackValue > 0) {
    ratioData.push({ name: "攻击", value: attackValue, itemStyle: { color: CHART_COLORS.blue } });
  }
  if (benignValue > 0) {
    ratioData.push({ name: "正常", value: benignValue, itemStyle: { color: CHART_COLORS.dark } });
  }
  if (!ratioData.length) {
    ratioData.push({
      name: "等待事件",
      value: 1,
      label: { show: false },
      labelLine: { show: false },
      itemStyle: { color: CHART_COLORS.track }
    });
  }

  return {
    animation: false,
    tooltip: {
      trigger: "item",
      backgroundColor: "rgba(255, 255, 255, 0.96)",
      borderColor: "rgba(29, 29, 31, 0.08)",
      textStyle: { color: CHART_COLORS.dark }
    },
    series: [
      {
        type: "pie",
        radius: ["54%", "76%"],
        center: ["50%", "50%"],
        stillShowZeroSum: false,
        label: {
          color: CHART_COLORS.dark,
          formatter: ({ name, percent }) => `${name}\n${percent.toFixed(1)}%`
        },
        labelLine: { lineStyle: { color: CHART_COLORS.track } },
        itemStyle: {
          borderColor: "#f5f5f7",
          borderWidth: 4
        },
        data: ratioData
      }
    ],
    graphic: [
      {
        type: "text",
        left: "center",
        top: "42%",
        style: {
          text: totalEvents.value ? formatCount(totalEvents.value) : "0",
          fontSize: 30,
          fontWeight: 600,
          fill: CHART_COLORS.dark
        }
      },
      {
        type: "text",
        left: "center",
        top: "56%",
        style: {
          text: "总事件",
          fontSize: 13,
          fill: CHART_COLORS.darkMuted
        }
      }
    ]
  };
}

function getDistributionOption() {
  const protocols = Object.entries(state.protocolCounts)
    .sort((left, right) => right[1] - left[1])
    .slice(0, 5)
    .reverse();
  const ports = Object.entries(state.portCounts)
    .sort((left, right) => right[1] - left[1])
    .slice(0, 5)
    .reverse();

  const axis = buildAxisStyle(false);

  if (!hasProtocolData.value) {
    return {
      animationDuration: 300,
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        backgroundColor: "rgba(255, 255, 255, 0.96)",
        borderColor: "rgba(29, 29, 31, 0.08)",
        textStyle: { color: CHART_COLORS.dark }
      },
      grid: { left: 22, right: 22, top: 38, bottom: 16, containLabel: true },
      xAxis: { type: "value", ...axis },
      yAxis: {
        type: "category",
        data: ports.map(([label]) => String(label || "-")),
        axisTick: { show: false },
        ...axis
      },
      series: [
        {
          name: "端口",
          type: "bar",
          barWidth: 10,
          data: ports.map(([, value]) => value),
          itemStyle: { color: CHART_COLORS.dark, borderRadius: [0, 6, 6, 0] }
        }
      ],
      graphic: [
        {
          type: "text",
          left: 28,
          top: 10,
          style: { text: "端口", fill: CHART_COLORS.darkMuted, fontSize: 12, fontWeight: 600 }
        }
      ]
    };
  }

  return {
    animationDuration: 300,
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      backgroundColor: "rgba(255, 255, 255, 0.96)",
      borderColor: "rgba(29, 29, 31, 0.08)",
      textStyle: { color: CHART_COLORS.dark }
    },
    grid: [
      { left: 22, top: 38, width: "38%", bottom: 16, containLabel: true },
      { right: 22, top: 38, width: "38%", bottom: 16, containLabel: true }
    ],
    xAxis: [
      { type: "value", gridIndex: 0, ...axis },
      { type: "value", gridIndex: 1, ...axis }
    ],
    yAxis: [
      {
        type: "category",
        gridIndex: 0,
        data: protocols.map(([label]) => label || "-"),
        axisTick: { show: false },
        ...axis
      },
      {
        type: "category",
        gridIndex: 1,
        data: ports.map(([label]) => String(label || "-")),
        axisTick: { show: false },
        ...axis
      }
    ],
    series: [
      {
        name: "协议",
        type: "bar",
        xAxisIndex: 0,
        yAxisIndex: 0,
        barWidth: 10,
        data: protocols.map(([, value]) => value),
        itemStyle: { color: CHART_COLORS.blue, borderRadius: [0, 6, 6, 0] }
      },
      {
        name: "端口",
        type: "bar",
        xAxisIndex: 1,
        yAxisIndex: 1,
        barWidth: 10,
        data: ports.map(([, value]) => value),
        itemStyle: { color: CHART_COLORS.dark, borderRadius: [0, 6, 6, 0] }
      }
    ],
    graphic: [
      {
        type: "text",
        left: 28,
        top: 10,
        style: { text: "协议", fill: CHART_COLORS.darkMuted, fontSize: 12, fontWeight: 600 }
      },
      {
        type: "text",
        right: 28,
        top: 10,
        style: { text: "端口", fill: CHART_COLORS.darkMuted, fontSize: 12, fontWeight: 600 }
      }
    ]
  };
}

function refreshCharts() {
  updateChart(throughputChart, getThroughputOption());
  updateChart(ratioChart, getRatioOption());
  updateChart(distributionChart, getDistributionOption());
}

function resizeCharts() {
  throughputChart?.resize();
  ratioChart?.resize();
  distributionChart?.resize();
}

function registerThroughputPoint(timestamp) {
  const key = new Date(timestamp || Date.now()).toLocaleTimeString("zh-CN", { hour12: false });
  const lastIndex = state.throughputTimes.length - 1;
  if (state.throughputTimes[lastIndex] === key) {
    state.throughputValues[lastIndex] += 1;
  } else {
    state.throughputTimes.push(key);
    state.throughputValues.push(1);
  }
  if (state.throughputTimes.length > 60) {
    state.throughputTimes.shift();
    state.throughputValues.shift();
  }
}

function ingestEvent(event) {
  if (!event || !rememberEvent(event)) {
    return;
  }

  const prediction = event.prediction || "benign";
  if (prediction === "attack") {
    state.attackCount += 1;
  } else {
    state.benignCount += 1;
  }

  const protocol = normalizeProtocol(event.display_fields?.protocol);
  const port = String(event.display_fields?.destination_port ?? "未记录");
  if (protocol) {
    state.protocolCounts[protocol] = (state.protocolCounts[protocol] || 0) + 1;
  }
  state.portCounts[port] = (state.portCounts[port] || 0) + 1;

  registerThroughputPoint(event.timestamp);

  state.currentFlow = event.payload || event;
  state.recentAlerts.unshift(event);
  if (state.recentAlerts.length > 100) {
    state.recentAlerts.pop();
  }
  refreshCharts();
}

function resetDashboard() {
  state.throughputTimes = [];
  state.throughputValues = [];
  state.benignCount = 0;
  state.attackCount = 0;
  state.protocolCounts = {};
  state.portCounts = {};
  state.recentAlerts = [];
  state.currentFlow = null;
  pollCursor = 0;
  seenEventKeys.clear();
  setActionMessage("面板已重置。");
  refreshCharts();
}

async function refreshHealth() {
  try {
    await getHealth();
    health.ok = true;
  } catch {
    health.ok = false;
  }
}

async function refreshReplayStatus() {
  try {
    const data = await getReplayStatus();
    replay.running = Boolean(data.status?.running ?? data.running);
  } catch {
    replay.running = false;
  }
}

async function handleStartReplay() {
  try {
    await startReplay({
      rate_per_second: Number(controls.ratePerSecond),
      batch_size: Number(controls.batchSize),
      max_events: 10000
    });
    await refreshReplayStatus();
    setActionMessage("回放已启动，事件正在进入看板。", "success");
  } catch (error) {
    setActionMessage(error?.message || "启动回放失败。", "error");
  }
}

async function handleStopReplay() {
  try {
    await stopReplay();
    await refreshReplayStatus();
    setActionMessage("已停止回放。", "success");
  } catch (error) {
    setActionMessage(error?.message || "停止回放失败。", "error");
  }
}

async function fallbackPollEvents() {
  if (wsConnected.value) {
    return;
  }
  try {
    const payload = await getReplayEvents(50, pollCursor);
    const events = Array.isArray(payload) ? payload : payload.events || [];
    if (!Array.isArray(payload) && typeof payload.next_cursor === "number") {
      pollCursor = payload.next_cursor;
    }
    for (const item of events) {
      ingestEvent(item);
    }
  } catch {
    return;
  }
}

function scheduleReconnect() {
  if (isUnmounted || reconnectTimer || wsConnected.value) {
    return;
  }

  reconnectTimer = window.setTimeout(() => {
    reconnectTimer = null;
    if (!isUnmounted && !wsConnected.value) {
      openSocket();
    }
  }, 4000);
}

function openSocket() {
  if (socket) {
    socket.close();
    socket = null;
  }

  socket = createAlertSocket(
    (payload) => {
      if (payload?.type === "status") {
        replay.running = Boolean(payload.status?.running);
        return;
      }

      const events =
        payload?.type === "alert" && payload.event
          ? [payload.event]
          : Array.isArray(payload)
            ? payload
            : payload?.events
              ? payload.events
              : [payload];

      for (const item of events) {
        if (item?.prediction || item?.event_id) {
          ingestEvent(item);
        }
      }
    },
    () => {
      wsConnected.value = true;
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    },
    () => {
      wsConnected.value = false;
      socket = null;
      scheduleReconnect();
    },
    () => {
      wsConnected.value = false;
    }
  );
}

onMounted(async () => {
  throughputChart = initChart(throughputRef.value, getThroughputOption());
  ratioChart = initChart(ratioRef.value, getRatioOption());
  distributionChart = initChart(distributionRef.value, getDistributionOption());

  await refreshHealth();
  await refreshReplayStatus();
  openSocket();

  pollTimer = window.setInterval(fallbackPollEvents, 3000);
  statusTimer = window.setInterval(async () => {
    await refreshHealth();
    await refreshReplayStatus();
  }, 5000);

  window.addEventListener("resize", resizeCharts);
});

onBeforeUnmount(() => {
  isUnmounted = true;

  if (pollTimer) {
    clearInterval(pollTimer);
  }
  if (statusTimer) {
    clearInterval(statusTimer);
  }
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
  }
  if (socket) {
    socket.close();
  }

  window.removeEventListener("resize", resizeCharts);
  throughputChart?.dispose();
  ratioChart?.dispose();
  distributionChart?.dispose();
});
</script>
