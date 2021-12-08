import React, {Component} from 'react';
import 'antd/dist/antd.css';
import { LineChart, Line, XAxis, YAxis, Legend, Tooltip } from 'recharts'
import { Route, Link, Switch } from 'react-router-dom';
import { Layout, Menu, Space, Button } from 'antd';
import { MenuFoldOutlined, MenuUnfoldOutlined } from '@ant-design/icons'

import Equipment from './EquipmentView';

const { Sider, Content, Header } = Layout


function StatusDot(props) {
    const colors = [
        "#00FF00",
        "#FFFF00",
        "#FF0000",
        "#AAAAAA"
    ]

    let status = props.status
    if (status < 0 || status > 2) {
        status = 3
    }

    return <svg width={15} height={15}>
        <circle r={5} cx={10} cy={10} fill={colors[status]}></circle>
    </svg>
}

function StatusWithName(props) {
    return <div>
        <StatusDot status={props.status} />
        <span style={{marginLeft: 5, fontWeight: "bold"}}>
            {props.name}
        </span>
    </div>
}

export default class Plant extends Component {
    constructor(props) {
        super(props);
        this.state = {
            steps: [],
            collapsed: false,
        };
    }

    fetchAndSetState() {
        fetch('/plant')
        .then((res) => res.json())
        .then((data) => this.setState(data));
    }

    componentDidMount() {
        this.updateIntervalID = setInterval(this.fetchAndSetState.bind(this), 2000);
    }

    componentWillUnmount() {
        clearInterval(this.updateIntervalID);
    }

    toggleSider = () => {
        this.setState({
            collapsed: !this.state.collapsed,
        })
    };

    reset = () => {
        fetch('/reset', {method: 'POST'});
    };

    render() {
        return (
            <Layout>
                <Sider
                    collapsible
                    collapsedWidth={0}
                    collapsed={this.state.collapsed}
                    trigger={null}
                    style={{
                        overflow: 'hidden',
                        height: '100vh',
                        position: 'fixed',
                        left: 0,
                        backgroundColor: "#FFFFFF"
                    }}
                >
                    <Button onClick={this.reset} style={{margin: '20px'}}>RESET BATCH</Button>    
                    <h4 style={{margin: '10px'}}>Anlage</h4>
                    <Menu mode='inline'>
                        {this.state.steps.map((stepProps, i) => (
                            <Menu.ItemGroup
                                key={stepProps.name}
                                title={stepProps.name}
                            >
                                {stepProps.equipments.map((props, i) => 
                                    <Menu.Item
                                        key={props.name}
                                    >
                                        <Link to={"/equipment/"+stepProps.name+"/"+props.name}>
                                            <StatusWithName name={props.name} status={props.status} />
                                        </Link>
                                    </Menu.Item>
                                )}
                            </Menu.ItemGroup>
                        ))}
                    </Menu>
                </Sider>
                <Layout
                    style={{
                        marginLeft: this.state.collapsed ? 20 : 200,
                        transition: "margin-left 0.25s"}}>
                    {React.createElement(this.state.collapsed ? MenuUnfoldOutlined : MenuFoldOutlined, {
                            className: 'trigger',
                            onClick: this.toggleSider,
                            style: {width: 15, marginLeft: 0},
                    })}
                    <Content>
                        <Switch>
                            <Route path='/equipment/:step/:name' render={(props) => <Equipment {...props} />} />
                        </Switch>
                    </Content>
                </Layout>
            </Layout>
        );
    }
}
