import React, { useState, useRef, useEffect } from 'react';
import * as THREE from 'three';
import * as math from 'mathjs';

const LorentzSpringNetwork = () => {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const nodesRef = useRef([]);
  const connectionsRef = useRef([]);
  const springGroupRef = useRef(null);
  const fieldGroupRef = useRef(null);
  const forceGroupRef = useRef(null);
  
  const [physicsActive, setPhysicsActive] = useState(false);
  const [showForces, setShowForces] = useState(true);
  const [showField, setShowField] = useState(false);
  const [springConstant, setSpringConstant] = useState(0.5);
  const [dampingFactor, setDampingFactor] = useState(0.95);
  
  // PESTLE factors drive electromagnetic field
  const [pestleFactors, setPestleFactors] = useState({
    political: 0.65,    // Electric field strength
    economic: 0.72,     // Magnetic field strength  
    social: 0.58,       // Charge density
    technological: 0.83, // Field frequency
    legal: 0.41,        // Field damping
    environmental: 0.35  // Field turbulence
  });

  const [metrics, setMetrics] = useState({
    totalEnergy: 0,
    springEnergy: 0,
    kineticEnergy: 0,
    magneticEnergy: 0,
    systemStability: 0,
    avgVelocity: 0
  });

  // Create nodes with wants/needs/prestige positioning
  const createNodes = () => {
    const nodeTypes = ['Government', 'Corporation', 'Academics', 'NGO', 'Media'];
    const nodes = [];
    
    for (let i = 0; i < 20; i++) {
      // Core attributes
      const wants = Math.random() * 20 - 10;      // X-axis: what they want
      const needs = Math.random() * 20 - 10;      // Y-axis: what they need  
      const prestige = Math.random() * 10 - 5;    // Z-axis: their status/prestige
      
      const node = {
        id: i,
        name: `Node_${i}`,
        type: nodeTypes[i % nodeTypes.length],
        
        // World placement coordinates
        wants: wants,
        needs: needs, 
        prestige: prestige,
        
        // Physics properties
        position: new THREE.Vector3(wants, needs, prestige),
        velocity: new THREE.Vector3(0, 0, 0),
        acceleration: new THREE.Vector3(0, 0, 0),
        mass: 0.5 + Math.random() * 1.5, // Variable mass
        charge: (Math.random() - 0.5) * 2, // Electric charge for Lorentz force
        
        // Relationship properties
        trust: Math.random() * 0.8 + 0.2,
        innovation: Math.random(),
        influence: Math.random() * 0.5,
        
        // Force tracking
        springForce: new THREE.Vector3(0, 0, 0),
        lorentzForce: new THREE.Vector3(0, 0, 0),
        totalForce: new THREE.Vector3(0, 0, 0),
        
        originalPosition: null
      };
      
      node.originalPosition = node.position.clone();
      nodes.push(node);
    }
    
    return nodes;
  };

  // Create spring connections based on compatibility
  const createConnections = (nodes) => {
    const connections = [];
    
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const nodeA = nodes[i];
        const nodeB = nodes[j];
        
        // Calculate compatibility in wants/needs/prestige space
        const wantsDiff = Math.abs(nodeA.wants - nodeB.wants);
        const needsDiff = Math.abs(nodeA.needs - nodeB.needs);
        const prestigeDiff = Math.abs(nodeA.prestige - nodeB.prestige);
        
        // Compatibility score (lower differences = higher compatibility)
        const compatibility = 1 / (1 + wantsDiff * 0.1 + needsDiff * 0.1 + prestigeDiff * 0.2);
        const trustFactor = (nodeA.trust + nodeB.trust) / 2;
        const connectionStrength = compatibility * trustFactor;
        
        // Create connection if strong enough
        if (connectionStrength > 0.4) {
          const distance = nodeA.position.distanceTo(nodeB.position);
          connections.push({
            nodeA: i,
            nodeB: j,
            strength: connectionStrength,
            restLength: distance * 0.8, // Preferred spring length
            currentLength: distance,
            springConstant: springConstant * connectionStrength,
            compatibility: compatibility
          });
        }
      }
    }
    
    return connections;
  };

  // Calculate electromagnetic field at position
  const calculateEMField = (position, time) => {
    const P = pestleFactors.political;    // E-field strength
    const E = pestleFactors.economic;     // B-field strength  
    const S = pestleFactors.social;       // Charge density
    const T = pestleFactors.technological; // Frequency
    const L = pestleFactors.legal;        // Damping
    const ENV = pestleFactors.environmental; // Turbulence
    
    const x = position.x, y = position.y, z = position.z;
    
    // Electric field (political factor drives strength)
    const Ex = P * Math.cos(T * time + x * 0.1) * Math.exp(-L * time * 0.01);
    const Ey = P * Math.sin(T * time + y * 0.1) * Math.exp(-L * time * 0.01);
    const Ez = P * Math.cos(T * time + z * 0.2) * Math.exp(-L * time * 0.01);
    const electricField = new THREE.Vector3(Ex, Ey, Ez);
    
    // Magnetic field (economic factor drives strength)
    const Bx = E * Math.sin(T * time * 0.7 + y * 0.1) + ENV * (Math.random() - 0.5) * 0.1;
    const By = E * Math.cos(T * time * 0.7 + x * 0.1) + ENV * (Math.random() - 0.5) * 0.1;
    const Bz = E * Math.sin(T * time * 0.7 + x * 0.05 + y * 0.05) + ENV * (Math.random() - 0.5) * 0.1;
    const magneticField = new THREE.Vector3(Bx, By, Bz);
    
    return { electricField, magneticField };
  };

  // Calculate Lorentz force: F = q(E + v × B)
  const calculateLorentzForce = (node, time) => {
    const { electricField, magneticField } = calculateEMField(node.position, time);
    
    // F = q(E + v × B)
    const electricForce = electricField.clone().multiplyScalar(node.charge);
    const velocityCrossMagnetic = new THREE.Vector3().crossVectors(node.velocity, magneticField);
    const magneticForce = velocityCrossMagnetic.multiplyScalar(node.charge);
    
    return electricForce.add(magneticForce);
  };

  // Calculate spring forces between connected nodes
  const calculateSpringForces = (nodes, connections) => {
    // Reset spring forces
    nodes.forEach(node => node.springForce.set(0, 0, 0));
    
    connections.forEach(conn => {
      const nodeA = nodes[conn.nodeA];
      const nodeB = nodes[conn.nodeB];
      
      // Current spring vector
      const springVector = new THREE.Vector3().subVectors(nodeB.position, nodeA.position);
      const currentLength = springVector.length();
      conn.currentLength = currentLength;
      
      if (currentLength > 0.001) {
        // Spring force: F = -k(x - x0)
        const displacement = currentLength - conn.restLength;
        const springDirection = springVector.normalize();
        const springMagnitude = conn.springConstant * displacement;
        
        const forceA = springDirection.clone().multiplyScalar(springMagnitude);
        const forceB = springDirection.clone().multiplyScalar(-springMagnitude);
        
        nodeA.springForce.add(forceA);
        nodeB.springForce.add(forceB);
      }
    });
  };

  // Apply F = ma physics
  const updatePhysics = (nodes, connections, time, dt) => {
    if (!physicsActive) return;
    
    // Calculate all forces
    calculateSpringForces(nodes, connections);
    
    nodes.forEach(node => {
      // Calculate Lorentz force
      node.lorentzForce = calculateLorentzForce(node, time);
      
      // Total force = spring + Lorentz + damping
      const dampingForce = node.velocity.clone().multiplyScalar(-0.1 * (1 - dampingFactor));
      node.totalForce = node.springForce.clone()
        .add(node.lorentzForce)
        .add(dampingForce);
      
      // F = ma -> a = F/m
      node.acceleration = node.totalForce.clone().divideScalar(node.mass);
      
      // Update velocity and position using Verlet integration
      node.velocity.add(node.acceleration.clone().multiplyScalar(dt));
      node.velocity.multiplyScalar(dampingFactor); // Global damping
      node.position.add(node.velocity.clone().multiplyScalar(dt));
    });
  };

  // Calculate system energy and metrics
  const calculateMetrics = (nodes, connections) => {
    let springEnergy = 0;
    let kineticEnergy = 0;
    let magneticEnergy = 0;
    let totalVelocity = 0;
    
    // Spring potential energy
    connections.forEach(conn => {
      const displacement = conn.currentLength - conn.restLength;
      springEnergy += 0.5 * conn.springConstant * displacement * displacement;
    });
    
    // Kinetic energy and velocities
    nodes.forEach(node => {
      const v2 = node.velocity.lengthSq();
      kineticEnergy += 0.5 * node.mass * v2;
      totalVelocity += Math.sqrt(v2);
      
      // Magnetic energy contribution
      magneticEnergy += 0.5 * node.charge * node.charge * node.lorentzForce.lengthSq();
    });
    
    const totalEnergy = springEnergy + kineticEnergy + magneticEnergy;
    const avgVelocity = totalVelocity / nodes.length;
    
    // System stability (lower energy = more stable)
    const systemStability = Math.exp(-totalEnergy / 10);
    
    setMetrics({
      totalEnergy: totalEnergy.toFixed(2),
      springEnergy: springEnergy.toFixed(2),
      kineticEnergy: kineticEnergy.toFixed(2),
      magneticEnergy: magneticEnergy.toFixed(2),
      systemStability: systemStability.toFixed(3),
      avgVelocity: avgVelocity.toFixed(3)
    });
  };

  // Visualize forces as arrows
  const visualizeForces = (scene, nodes) => {
    if (!forceGroupRef.current) return;
    forceGroupRef.current.clear();
    
    if (!showForces) return;
    
    nodes.forEach(node => {
      // Spring force arrows (green)
      if (node.springForce.length() > 0.1) {
        const springArrow = new THREE.ArrowHelper(
          node.springForce.clone().normalize(),
          node.position,
          Math.min(node.springForce.length() * 2, 3),
          0x00ff00,
          0.3,
          0.2
        );
        springArrow.line.material.transparent = true;
        springArrow.line.material.opacity = 0.7;
        forceGroupRef.current.add(springArrow);
      }
      
      // Lorentz force arrows (red)
      if (node.lorentzForce.length() > 0.1) {
        const lorentzArrow = new THREE.ArrowHelper(
          node.lorentzForce.clone().normalize(),
          node.position.clone().add(new THREE.Vector3(0.5, 0, 0)),
          Math.min(node.lorentzForce.length() * 2, 3),
          0xff0000,
          0.3,
          0.2
        );
        lorentzArrow.line.material.transparent = true;
        lorentzArrow.line.material.opacity = 0.7;
        forceGroupRef.current.add(lorentzArrow);
      }
    });
  };

  // Visualize electromagnetic field
  const visualizeEMField = (scene, time) => {
    if (!fieldGroupRef.current) return;
    fieldGroupRef.current.clear();
    
    if (!showField) return;
    
    const resolution = 3;
    const range = 8;
    
    for (let x = -range; x <= range; x += range/resolution) {
      for (let y = -range; y <= range; y += range/resolution) {
        const position = new THREE.Vector3(x, y, 0);
        const { electricField, magneticField } = calculateEMField(position, time);
        
        // Electric field vectors (blue)
        if (electricField.length() > 0.1) {
          const eArrow = new THREE.ArrowHelper(
            electricField.normalize(),
            position,
            Math.min(electricField.length(), 1.5),
            0x0088ff,
            0.2,
            0.1
          );
          eArrow.line.material.transparent = true;
          eArrow.line.material.opacity = 0.4;
          fieldGroupRef.current.add(eArrow);
        }
        
        // Magnetic field vectors (magenta)
        if (magneticField.length() > 0.1) {
          const bArrow = new THREE.ArrowHelper(
            magneticField.normalize(),
            position.clone().add(new THREE.Vector3(0.3, 0.3, 0)),
            Math.min(magneticField.length(), 1.5),
            0xff00ff,
            0.2,
            0.1
          );
          bArrow.line.material.transparent = true;
          bArrow.line.material.opacity = 0.4;
          fieldGroupRef.current.add(bArrow);
        }
      }
    }
  };

  const updatePestleFactor = (factor, value) => {
    setPestleFactors(prev => ({
      ...prev,
      [factor]: value
    }));
  };

  const resetSystem = () => {
    nodesRef.current.forEach(node => {
      node.position.copy(node.originalPosition);
      node.velocity.set(0, 0, 0);
      node.acceleration.set(0, 0, 0);
      node.springForce.set(0, 0, 0);
      node.lorentzForce.set(0, 0, 0);
    });
  };

  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
    
    const camera = new THREE.PerspectiveCamera(75, 900/700, 0.1, 1000);
    camera.position.set(15, 10, 25);
    
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(900, 700);
    mountRef.current.appendChild(renderer.domElement);
    
    sceneRef.current = scene;

    // Initialize groups
    const springGroup = new THREE.Group();
    const fieldGroup = new THREE.Group();
    const forceGroup = new THREE.Group();
    
    scene.add(springGroup, fieldGroup, forceGroup);
    
    springGroupRef.current = springGroup;
    fieldGroupRef.current = fieldGroup;
    forceGroupRef.current = forceGroup;

    // Create coordinate system axes
    const axesHelper = new THREE.AxesHelper(10);
    scene.add(axesHelper);
    
    // Add labels for axes
    const createAxisLabel = (text, position, color) => {
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      canvas.width = 128;
      canvas.height = 64;
      context.fillStyle = color;
      context.font = '24px Arial';
      context.fillText(text, 10, 40);
      
      const texture = new THREE.CanvasTexture(canvas);
      const material = new THREE.SpriteMaterial({ map: texture });
      const sprite = new THREE.Sprite(material);
      sprite.position.copy(position);
      sprite.scale.set(2, 1, 1);
      scene.add(sprite);
    };
    
    createAxisLabel('Wants', new THREE.Vector3(12, 0, 0), '#ff4444');
    createAxisLabel('Needs', new THREE.Vector3(0, 12, 0), '#44ff44');
    createAxisLabel('Prestige', new THREE.Vector3(0, 0, 8), '#4444ff');

    // Create nodes and connections
    const nodes = createNodes();
    const connections = createConnections(nodes);
    
    nodesRef.current = nodes;
    connectionsRef.current = connections;

    // Visualize nodes with charge-based coloring
    nodes.forEach(node => {
      const geometry = new THREE.SphereGeometry(0.3 + node.mass * 0.2, 16, 16);
      
      // Color based on charge and type
      let baseColor;
      if (node.type === 'Government') baseColor = 0xff6b6b;
      else if (node.type === 'Corporation') baseColor = 0x4ecdc4;
      else if (node.type === 'Academics') baseColor = 0x45b7d1;
      else if (node.type === 'NGO') baseColor = 0x96ceb4;
      else baseColor = 0xfeca57;
      
      // Modulate color by charge
      const color = new THREE.Color(baseColor);
      if (node.charge > 0) {
        color.lerp(new THREE.Color(0xffffff), Math.abs(node.charge) * 0.3);
      } else {
        color.lerp(new THREE.Color(0x000000), Math.abs(node.charge) * 0.3);
      }
      
      const material = new THREE.MeshBasicMaterial({ 
        color: color, 
        transparent: true, 
        opacity: 0.8 
      });
      
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.copy(node.position);
      mesh.userData = { nodeId: node.id };
      
      scene.add(mesh);
      node.mesh = mesh;
    });

    // Visualize spring connections
    connections.forEach(conn => {
      const nodeA = nodes[conn.nodeA];
      const nodeB = nodes[conn.nodeB];
      
      const geometry = new THREE.BufferGeometry();
      const positions = new Float32Array([
        nodeA.position.x, nodeA.position.y, nodeA.position.z,
        nodeB.position.x, nodeB.position.y, nodeB.position.z
      ]);
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      
      const material = new THREE.LineBasicMaterial({ 
        color: 0x888888, 
        transparent: true, 
        opacity: 0.4 + conn.strength * 0.4
      });
      
      const line = new THREE.Line(geometry, material);
      springGroup.add(line);
      conn.line = line;
    });

    // Animation loop
    let time = 0;
    const animate = () => {
      time += 0.016;
      
      // Update physics
      updatePhysics(nodes, connections, time, 0.016);
      
      // Update node meshes
      nodes.forEach(node => {
        if (node.mesh) {
          node.mesh.position.copy(node.position);
          
          // Scale based on velocity magnitude
          const velocityMag = node.velocity.length();
          const scale = 1 + velocityMag * 0.2;
          node.mesh.scale.setScalar(scale);
        }
      });
      
      // Update spring connections
      connections.forEach(conn => {
        if (conn.line) {
          const nodeA = nodes[conn.nodeA];
          const nodeB = nodes[conn.nodeB];
          const positions = conn.line.geometry.attributes.position;
          
          positions.setXYZ(0, nodeA.position.x, nodeA.position.y, nodeA.position.z);
          positions.setXYZ(1, nodeB.position.x, nodeB.position.y, nodeB.position.z);
          positions.needsUpdate = true;
          
          // Color spring based on tension
          const tension = Math.abs(conn.currentLength - conn.restLength) / conn.restLength;
          const color = new THREE.Color().setHSL(Math.max(0, 0.3 - tension), 0.8, 0.5);
          conn.line.material.color = color;
        }
      });
      
      // Visualize forces and fields
      visualizeForces(scene, nodes);
      visualizeEMField(scene, time);
      
      // Calculate metrics
      if (time % 30 === 0) { // Every 0.5 seconds
        calculateMetrics(nodes, connections);
      }
      
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    };
    
    animate();

    return () => {
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [physicsActive, showForces, showField, springConstant, dampingFactor, pestleFactors]);

  return (
    <div className="w-full bg-slate-900 text-white p-6 rounded-lg">
      <div className="mb-6">
        <h1 className="text-2xl font-bold mb-2">F=ma + Lorentz + Spring Network</h1>
        <p className="text-slate-300">Physics-based network in wants/needs/prestige space</p>
      </div>
      
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        {/* Visualization */}
        <div className="xl:col-span-3">
          <div ref={mountRef} className="border border-slate-700 rounded-lg overflow-hidden" />
          <div className="mt-2 text-xs text-slate-400 grid grid-cols-3 gap-4">
            <div><span className="text-red-400">X-axis:</span> Wants</div>
            <div><span className="text-green-400">Y-axis:</span> Needs</div>
            <div><span className="text-blue-400">Z-axis:</span> Prestige</div>
          </div>
        </div>
        
        {/* Controls */}
        <div className="space-y-4">
          {/* Physics Controls */}
          <div className="bg-slate-800 p-4 rounded-lg">
            <h3 className="text-lg font-semibold mb-3">Physics Engine</h3>
            
            <div className="space-y-3">
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="physics-active"
                  checked={physicsActive}
                  onChange={(e) => setPhysicsActive(e.target.checked)}
                />
                <label htmlFor="physics-active">Activate Physics</label>
              </div>
              
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="show-forces"
                  checked={showForces}
                  onChange={(e) => setShowForces(e.target.checked)}
                />
                <label htmlFor="show-forces">Show Forces</label>
              </div>
              
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="show-field"
                  checked={showField}
                  onChange={(e) => setShowField(e.target.checked)}
                />
                <label htmlFor="show-field">Show EM Field</label>
              </div>
              
              <button 
                onClick={resetSystem}
                className="w-full px-3 py-2 bg-red-600 hover:bg-red-700 rounded text-sm"
              >
                Reset System
              </button>
            </div>
          </div>
          
          {/* Spring Parameters */}
          <div className="bg-slate-800 p-4 rounded-lg">
            <h3 className="text-base font-semibold mb-3">Spring Physics</h3>
            
            <div className="space-y-3">
              <div>
                <label className="block text-sm mb-1">
                  Spring Constant: {springConstant.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="2.0"
                  step="0.1"
                  value={springConstant}
                  onChange={(e) => setSpringConstant(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
              
              <div>
                <label className="block text-sm mb-1">
                  Damping: {dampingFactor.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0.80"
                  max="0.99"
                  step="0.01"
                  value={dampingFactor}
                  onChange={(e) => setDampingFactor(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>
          </div>
          
          {/* PESTLE Electromagnetic Field */}
          <div className="bg-slate-800 p-4 rounded-lg">
            <h3 className="text-base font-semibold mb-3">EM Field (PESTLE)</h3>
            
            <div className="space-y-2 text-sm">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-red-400">Political (E-field)</span>
                  <span>{pestleFactors.political.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={pestleFactors.political}
                  onChange={(e) => updatePestleFactor('political', parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
              
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-teal-400">Economic (B-field)</span>
                  <span>{pestleFactors.economic.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={pestleFactors.economic}
                  onChange={(e) => updatePestleFactor('economic', parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
              
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-blue-400">Social (Charge)</span>
                  <span>{pestleFactors.social.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={pestleFactors.social}
                  onChange={(e) => updatePestleFactor('social', parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
              
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-green-400">Tech (Frequency)</span>
                  <span>{pestleFactors.technological.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={pestleFactors.technological}
                  onChange={(e) => updatePestleFactor('technological', parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>
          </div>
          
          {/* System Metrics */}
          <div className="bg-slate-800 p-4 rounded-lg">
            <h3 className="text-base font-semibold mb-3">System Metrics</h3>
            <div className="grid grid-cols-1 gap-2 text-xs">
              <div className="bg-slate-700 p-2 rounded">
                <div className="text-slate-400">Total Energy</div>
                <div className="font-mono text-white">{metrics.totalEnergy}</div>
              </div>
              <div className="bg-slate-700 p-2 rounded">
                <div className="text-slate-400">Spring Energy</div>
                <div className="font-mono text-green-400">{metrics.springEnergy}</div>
              </div>
              <div className="bg-slate-700 p-2 rounded">
                <div className="text-slate-400">Kinetic Energy</div>
                <div className="font-mono text-blue-400">{metrics.kineticEnergy}</div>
              </div>
              <div className="bg-slate-700 p-2 rounded">
                <div className="text-slate-400">Magnetic Energy</div>
                <div className="font-mono text-purple-400">{metrics.magneticEnergy}</div>
              </div>
              <div className="bg-slate-700 p-2 rounded">
                <div className="text-slate-400">Stability</div>
                <div className="font-mono text-orange-400">{metrics.systemStability}</div>
              </div>
              <div className="bg-slate-700 p-2 rounded">
                <div className="text-slate-400">Avg Velocity</div>
                <div className="font-mono text-cyan-400">{metrics.avgVelocity}</div>
              </div>
            </div>
          </div>
          
          {/* Force Legend */}
          <div className="bg-slate-800 p-3 rounded-lg">
            <h3 className="text-sm font-semibold mb-2">Force Legend</h3>
            <div className="space-y-1 text-xs">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-1 bg-green-500"></div>
                <span>Spring Forces</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-1 bg-red-500"></div>
                <span>Lorentz Forces</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-1 bg-blue-500"></div>
                <span>E-Field</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-1 bg-purple-500"></div>
                <span>B-Field</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LorentzSpringNetwork;
