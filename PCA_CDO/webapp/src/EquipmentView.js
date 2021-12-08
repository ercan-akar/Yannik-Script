import React, { Component } from 'react';
import logo from './logo.svg';
import { Grid, Row, Col } from 'react-flexbox-grid';
import { 
  LineChart,
  Line,
  XAxis,
  YAxis,
  Legend,
  Bar,
  BarChart,
  Tooltip
} from 'recharts';
import { SelectableDot, MeanStd3Chart, SelectableBarChart } from './Plot.js';
import DropDown from 'react-dropdown';
import 'react-dropdown/style.css';

// const colors = []
// for (let i = 0; i < 100; i++) {
//   colors.push("#" + Math.floor(Math.random()*16777215).toString(16))
// }

export default class EquipmentView extends Component {

  constructor(props) {
    super(props);
    this.state = {
      time_step: 0,
      time_stamps: [],
      n_time_steps: 0,
      variables: {},
      current_t: 0,
      current_batch_id: "",
      scores: {},
      t2: [],
      t2_contrib: {},
      t2_contrib_composed: {},
      dmodx: [],
      dmodx_contrib: [],
      selectedVariable: "",
      selectedScore: "",
      selectedTime: -1,
      selectedT2Weight: ""
    };
    
    this.plotWidth = 500;
    this.plotHeight = 300;
  }

  resetState() {
    this.setState({
      time_step: 0,
      time_stamps: [],
      n_time_steps: 0,
      variables: {},
      current_t: 0,
      current_batch_id: "",
      scores: {},
      t2: [],
      t2_contrib: {},
      t2_contrib_composed: {},
      dmodx: [],
      dmodx_contrib: [],
      selectedVariable: "",
      selectedScore: "",
      selectedTime: -1,
      selectedT2Weight: ""
    });
  }

  componentDidUpdate(oldProps) {
    if (oldProps.match.params.step != this.props.match.params.step || oldProps.match.params.name != this.props.match.params.name) {
      this.resetState();
      this.fetchAndSetState();    
    }
  }

  componentWillMount() {
    this.updateIntervalID = setInterval(this.fetchAndSetState.bind(this), 3000);
  }

  componentWillUnmount() {
    clearInterval(this.updateIntervalID);
  }

  fetchAndSetState() {
    fetch(`/equipment/${this.props.match.params.step}/${this.props.match.params.name}`)
    .then((res) => res.json())
    .then((state) => {
      this.setState(state);
      // set default values for dropdown
      const variableNames = Object.keys(this.state.variables);
      const scoreNames = Object.keys(this.state.scores);
      const weightNames = Object.keys(this.state.t2_contrib);
      this.setState({
        selectedVariable: this.state.selectedVariable || variableNames[0],
        selectedScore: this.state.selectedScore || scoreNames[0],
        selectedT2Weight: this.state.selectedT2Weight || weightNames[0]
      });
    });
  }

  onBarSelect = (data, index) => {
    this.onVariableSelect(this.variableNames[index]);
  }

  onVariableSelect = (selection) => {
    console.log("Selected: ", selection);
    this.setState({selectedVariable: selection});
  }

  onScoreSelect = (selection) => {
    this.setState({selectedScore: selection});
  }

  onT2WeightSelect = (selection) => {
    this.setState({selectedT2Weight: selection});
  }

  onTimeSelect = (idx) => {
    this.setState({selectedTime: idx});
  }

  render() {

    const variableNames = Object.keys(this.state.variables);
    const scoreNames = Object.keys(this.state.scores);
    const t2WeightNames = Object.keys(this.state.t2_contrib);

    return (
        <Grid fluid>
          <Row>
            <h1>Step: {this.props.match.params.step}, Equipment: {this.props.match.params.name}</h1>
          </Row>
          <Row>
            <Col>
              <h2>Hotellings T2</h2>
              <Row>
                <LineChart
                  data={this.state.t2}
                  width={this.plotWidth}
                  height={this.plotHeight}
                  >
                  <XAxis dataKey="t" />
                  <YAxis />
                  <Legend />
                  <Line
                    dataKey="value"
                    stroke="#000000"
                    animationDuration={0}
                    dot={SelectableDot(this.onTimeSelect, this.state.selectedTime)}
                    />
                  <Line 
                    dataKey="Crit95"
                    stroke="#FFFF00"
                    strokeWidth={2}
                    strokeDasharray="2 2"
                    dot={false}/>
                  <Line 
                    dataKey="Crit99"
                    stroke="#FF0000"
                    strokeWidth={2}
                    strokeDasharray="2 2"
                    dot={false}/>
                </LineChart>
              </Row>
            </Col>
            <Col>
              <Row>
                <Col><h2>Contributions</h2></Col>
                <Col><DropDown options={t2WeightNames} value={this.state.selectedT2Weight} onChange={(o) => this.onT2WeightSelect(o.value)} /></Col>
              </Row>
              <Row>
                <SelectableBarChart
                  width={this.plotWidth}
                  height={this.plotHeight}
                  data={this.state.selectedTime > -1 ? this.state.t2_contrib[this.state.selectedT2Weight][this.state.selectedTime] : []}
                  onSelect={this.onVariableSelect}
                  selected={this.state.selectedVariable}
                  />
              </Row>
            </Col>
            {/* <Col>
              <Row>
                <h2>Hotellings T2 composed</h2>
              </Row>
              <Row>
                <LineChart
                  data={this.state.t2_contrib_composed[this.state.selectedT2Weight]}
                  width={this.plotWidth}
                  height={this.plotHeight}
                  >
                  <XAxis dataKey="t" />
                  <YAxis />
                  <Legend />
                  <Line
                    dataKey="value"
                    stroke="#000000"
                    animationDuration={0}
                  />
                </LineChart>
              </Row>
            </Col> */}
          </Row>
          <Row>
            <Col>
              <h2>DModX</h2>
              <Row>
                <MeanStd3Chart
                  data={this.state.dmodx}
                  width={this.plotWidth}
                  height={this.plotHeight}
                  dot={SelectableDot(this.onTimeSelect, this.state.selectedTime)}
                  />
              </Row>
            </Col>
            <Col>
              <Row>
                <h2>Contributions</h2>
              </Row>
              <Row>
                <SelectableBarChart
                  width={this.plotWidth}
                  height={this.plotHeight}
                  data={this.state.selectedTime > -1 ? this.state.dmodx_contrib[this.state.selectedTime] : []}
                  onSelect={this.onVariableSelect}
                  selected={this.state.selectedVariable}
                  />
              </Row>
            </Col>
          </Row>
          <Row>
            <Col>
              <Row>
                <DropDown options={variableNames} value={this.state.selectedVariable} onChange={(o) => this.onVariableSelect(o.value)} />
              </Row>
              <Row>
                <MeanStd3Chart 
                  data={this.state.variables[this.state.selectedVariable]} 
                  dot={SelectableDot(this.onTimeSelect, this.state.selectedTime)}
                  />
              </Row> 
            </Col>
            <Col>
              <Row>
                <DropDown options={scoreNames} value={this.state.selectedScore} onChange={(o) => this.onScoreSelect(o.value)} />
              </Row>
              <Row>
                <MeanStd3Chart
                  data={this.state.scores[this.state.selectedScore]}
                  dot={SelectableDot(this.onTimeSelect, this.state.selectedTime)}
                  />
              </Row>
            </Col>
          </Row>
        </Grid>
    );
  }
}
