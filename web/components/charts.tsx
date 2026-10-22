"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

export const COLORS = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "var(--chart-5)",
  "var(--chart-6)",
];

type Row = Record<string, unknown>;
type Format = (value: number) => string;

export interface Series {
  key: string;
  label: string;
  color?: string;
  /** Plot on a second y axis on the right. */
  right?: boolean;
}

const axis = {
  tick: { fill: "var(--muted)", fontSize: 11, fontFamily: "var(--font-geist-mono)" },
  stroke: "var(--border)",
  tickLine: false,
};

/** Category labels read better in the UI font; numbers stay monospaced. */
const categoryTick = { ...axis.tick, fontFamily: "var(--font-geist-sans)" };

const tooltip = {
  contentStyle: {
    background: "var(--surface-2)",
    border: "1px solid var(--border-strong)",
    borderRadius: 10,
    fontSize: 12,
    color: "var(--fg)",
    boxShadow: "0 12px 32px -12px rgba(0, 0, 0, 0.8)",
  },
  labelStyle: { color: "var(--muted)", marginBottom: 4 },
  itemStyle: { padding: 0 },
  cursor: { fill: "rgba(255, 255, 255, 0.03)", stroke: "var(--border-strong)" },
};

const plain = (v: unknown) => String(v);

function formatValue(value: unknown, format: Format) {
  return typeof value === "number" ? format(value) : String(value ?? "");
}

export function TrendChart({
  data,
  x,
  series,
  height = 260,
  area = false,
  xFormat = plain,
  yFormat = plain,
  rightFormat = plain,
  bands = [],
}: {
  data: Row[];
  x: string;
  series: Series[];
  height?: number;
  area?: boolean;
  xFormat?: (value: string) => string;
  yFormat?: Format;
  rightFormat?: Format;
  bands?: { from: string; to: string; label: string }[];
}) {
  const hasRight = series.some((s) => s.right);
  const Chart = area ? AreaChart : LineChart;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <Chart data={data} margin={{ top: 8, right: hasRight ? 0 : 8, bottom: 0, left: 0 }}>
        <CartesianGrid stroke="var(--grid)" strokeDasharray="3 4" vertical={false} />
        <XAxis dataKey={x} {...axis} tickFormatter={xFormat} minTickGap={28} />
        <YAxis yAxisId="left" {...axis} tickFormatter={yFormat} width={56} />
        {hasRight && (
          <YAxis yAxisId="right" orientation="right" {...axis} tickFormatter={rightFormat} width={48} />
        )}
        <Tooltip
          {...tooltip}
          labelFormatter={(label) => xFormat(String(label))}
          formatter={(value, name, item) => {
            const s = series.find((candidate) => candidate.key === item.dataKey);
            return [formatValue(value, s?.right ? rightFormat : yFormat), name];
          }}
        />
        {series.length > 1 && <Legend wrapperStyle={{ fontSize: 12 }} iconType="plainline" />}
        {bands.map((band) => (
          <ReferenceArea
            key={band.label}
            yAxisId="left"
            x1={band.from}
            x2={band.to}
            fill="var(--warn)"
            fillOpacity={0.1}
            label={{ value: band.label, position: "insideTop", fill: "var(--muted)", fontSize: 11 }}
          />
        ))}
        {series.map((s, i) =>
          area ? (
            <Area
              isAnimationActive={false}
              key={s.key}
              yAxisId={s.right ? "right" : "left"}
              type="monotone"
              dataKey={s.key}
              name={s.label}
              stroke={s.color ?? COLORS[i]}
              fill={s.color ?? COLORS[i]}
              fillOpacity={0.1}
              strokeWidth={2}
              dot={false}
            />
          ) : (
            <Line
              isAnimationActive={false}
              key={s.key}
              yAxisId={s.right ? "right" : "left"}
              type="monotone"
              dataKey={s.key}
              name={s.label}
              stroke={s.color ?? COLORS[i]}
              strokeWidth={2}
              dot={false}
            />
          ),
        )}
      </Chart>
    </ResponsiveContainer>
  );
}

export function BarsChart({
  data,
  x,
  series,
  height = 260,
  horizontal = false,
  xFormat = plain,
  yFormat = plain,
  categoryWidth = 120,
  colorFor,
  zeroLine = false,
}: {
  data: Row[];
  x: string;
  series: Series[];
  height?: number;
  /** Categories on the y axis, bars growing to the right. */
  horizontal?: boolean;
  xFormat?: (value: string) => string;
  yFormat?: Format;
  categoryWidth?: number;
  /** Per-bar colour for a single series (for example, positive vs negative). */
  colorFor?: (row: Row) => string;
  zeroLine?: boolean;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={data}
        layout={horizontal ? "vertical" : "horizontal"}
        margin={{ top: 8, right: 12, bottom: 0, left: 0 }}
      >
        <CartesianGrid stroke="var(--grid)" strokeDasharray="3 4" vertical={horizontal} horizontal={!horizontal} />
        {horizontal ? (
          <>
            <XAxis type="number" {...axis} tickFormatter={yFormat} />
            <YAxis
              type="category"
              dataKey={x}
              {...axis}
              tick={categoryTick}
              width={categoryWidth}
              tickFormatter={xFormat}
              interval={0}
            />
          </>
        ) : (
          <>
            <XAxis dataKey={x} {...axis} tick={categoryTick} tickFormatter={xFormat} interval="preserveStartEnd" />
            <YAxis {...axis} tickFormatter={yFormat} width={56} />
          </>
        )}
        <Tooltip
          {...tooltip}
          labelFormatter={(label) => xFormat(String(label))}
          formatter={(value, name) => [formatValue(value, yFormat), name]}
        />
        {series.length > 1 && <Legend wrapperStyle={{ fontSize: 12 }} />}
        {zeroLine && (horizontal ? <ReferenceLine x={0} stroke="var(--muted)" /> : <ReferenceLine y={0} stroke="var(--muted)" />)}
        {series.map((s, i) => (
          <Bar
            isAnimationActive={false}
            key={s.key}
            dataKey={s.key}
            name={s.label}
            fill={s.color ?? COLORS[i]}
            radius={horizontal ? [0, 4, 4, 0] : [4, 4, 0, 0]}
            maxBarSize={36}
          >
            {colorFor && series.length === 1 && data.map((row, j) => <Cell key={j} fill={colorFor(row)} />)}
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function BubbleChart({
  data,
  x,
  y,
  size,
  label,
  xLabel,
  yLabel,
  xFormat = plain,
  yFormat = plain,
  height = 280,
}: {
  data: Row[];
  x: string;
  y: string;
  size: string;
  label: string;
  xLabel: string;
  yLabel: string;
  xFormat?: Format;
  yFormat?: Format;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 12, right: 16, bottom: 16, left: 0 }}>
        <CartesianGrid stroke="var(--grid)" strokeDasharray="3 4" />
        <XAxis
          type="number"
          dataKey={x}
          name={xLabel}
          {...axis}
          tickFormatter={xFormat}
          domain={["auto", "auto"]}
          label={{ value: xLabel, position: "insideBottom", offset: -8, fill: "var(--muted)", fontSize: 11 }}
        />
        <YAxis type="number" dataKey={y} name={yLabel} {...axis} tickFormatter={yFormat} width={56} domain={["auto", "auto"]} />
        <ZAxis type="number" dataKey={size} range={[80, 900]} />
        <Tooltip
          {...tooltip}
          content={({ payload }) => {
            const row = payload?.[0]?.payload as Row | undefined;
            if (!row) return null;
            return (
              <div className="rounded-lg border border-border bg-surface px-3 py-2 text-xs">
                <div className="mb-1 font-medium">{String(row[label])}</div>
                <div className="text-muted">
                  {xLabel}: {xFormat(Number(row[x]))}
                </div>
                <div className="text-muted">
                  {yLabel}: {yFormat(Number(row[y]))}
                </div>
              </div>
            );
          }}
        />
        <Scatter data={data} fillOpacity={0.75} isAnimationActive={false}>
          {data.map((_, i) => (
            <Cell key={i} fill={COLORS[i % COLORS.length]} />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}
