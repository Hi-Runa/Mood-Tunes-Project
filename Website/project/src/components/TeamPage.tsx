/*
  Ayush Vupalanchi, Vaibhav Alaparthi, Hiruna Devadithya
  1/23/25

  This file shows team member information with their photos and names.
*/

import React from 'react';

// Define the structure of a team member object
interface TeamMember {
  name: string;
  image: string;
}

// Array of team members with their names and image URLs
const teamMembers: TeamMember[] = [
  {
    name: "Vaibhav Alaparthi",
    image: "https://cdn.discordapp.com/attachments/1333924617659879526/1334035742967468062/image.jpg?ex=679b111a&is=6799bf9a&hm=06725a687d3dc525568fb1b66da677bfcfcad994831a8b5df6c512d16120800d&"
  },
  {
    name: "Hiruna Devadithya",
    image: "https://cdn.discordapp.com/attachments/1333924617659879526/1334027960709025853/image.png?ex=679b09da&is=6799b85a&hm=a31606c866d33feae7e8e66b1e7c8c69ab89764e2c9d6864e18733cc9f5f492a&"
  },
  {
    name: "Ayush Vupalanchi",
    image: "https://cdn.discordapp.com/attachments/1333924617659879526/1334034480083767368/image.png?ex=679b0fed&is=6799be6d&hm=73976a95146bb2d24b4e2523675a4f03491d7ca27ad52a25c91b529bd953936e&"
  }
];

// Component to display the team page
export function TeamPage() {
  return (
    <div className="bg-white py-24 sm:py-32">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
  <div className="mx-auto max-w-2xl lg:max-w-4xl">
    {/* Page title */}
    <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">Meet our team</h2>
    <p className="mt-6 text-lg leading-8 text-gray-600"></p>
      We're a dynamic group of individuals who are passionate about what we do and dedicated to delivering the best music recommendations for your mood.
    </p>
    
    {/* List of team members */}
    <div className="mt-20 space-y-20">
      {teamMembers.map((member) => (
        <div key={member.name} className="relative flex flex-col gap-8 sm:flex-row">
    <div className="sm:w-1/3">
      {/* Team member image */}
      <img
        src={member.image}
        alt={member.name}
        className="aspect-[4/5] w-full rounded-2xl object-cover"
      />
    </div>
    <div className="sm:w-2/3 sm:pl-8">
      {/* Team member name */}
      <h3 className="text-2xl font-semibold leading-8 tracking-tight text-gray-900">{member.name}</h3>
    </div>
        </div>
      ))}
    </div>
  </div>
      </div>
    </div>
  );
}