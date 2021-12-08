import React, { useState } from 'react'
import { 
    LineChart, 
    Line, 
    XAxis, 
    YAxis, 
    Tooltip, 
    Legend, 
    Cell,
    Bar,
    BarChart,
} from 'recharts';

export const SelectableDot = (callback, selectedIdx) => ((props) => {
    const {cx, cy, stroke, payload, value} = props;

    const fill = (selectedIdx === payload['idx']) ? "black" : "white";
    const r = 3;
    return (
            <circle key={payload['idx']} cx={cx} cy={cy} r={r} fill={fill} stroke={stroke} strokeWidth="1" onClick={()=>callback(payload['idx'])}/>
    );
});

export function SelectableBarChart(props) {
    return <BarChart
            width={props.width}
            height={props.height}
            data={props.data}>
            <XAxis dataKey="name" interval={0}/>
            <YAxis />
            <Bar dataKey="value" onClick={(d, i) => props.onSelect(d.name)}>
                {props.data ? props.data.map((entry, i) => 
                    <Cell 
                        cursor="pointer" 
                        fill={entry.name == props.selected ? "black" : "white"}
                        stroke="black" 
                        key={entry.name}/>
                ): null}
            </Bar>
        </BarChart>
}

export function MeanStd3Chart(props) {
        return (<div>
    <LineChart
        data={props.data}
        width={props.width || 400}
        height={props.height || 300}
        isAnimationActive={false}
        >
        <XAxis dataKey="t" />
        <YAxis />
        <Legend />
        <Line
            dataKey="value"
            stroke="#000000"
            dot={props.dot}
            animationDuration={0}
            />
        <Line
            dataKey="mean"
            stroke="#00FF00"
            strokeWidth={2}
            dot={false}
            />
        <Line
            dataKey="std+3"
            stroke="#FF0000"
            strokeWidth={2}
            strokeDasharray="2 2"
            dot={false}/>
        <Line
            dataKey="std-3"
            stroke="#FF0000"
            strokeWidth={2}
            strokeDasharray="2 2"
            dot={false}/>
    </LineChart></div>);
}
