import * as echarts from "echarts";

export function initChart(container, option) {
  if (!container) {
    return null;
  }
  const chart = echarts.init(container);
  chart.setOption(option);
  return chart;
}

export function updateChart(chart, option) {
  if (!chart) {
    return;
  }
  chart.setOption(option, { notMerge: false });
}

