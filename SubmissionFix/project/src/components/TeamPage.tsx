import React from 'react';

interface TeamMember {
  name: string;
  image: string;
}

const teamMembers: TeamMember[] = [
  {
    name: "Vaibhav Alaparthi",
    image: "https://images.unsplash.com/photo-1706325640622-7a0a11f5d89f"
  },
  {
    name: "Hiruna Devaditya",
    image: "https://images.unsplash.com/photo-1706325640621-7a0a11f5d90f"
  },
  {
    name: "Ayush Vupalanchi",
    image: "https://images.unsplash.com/photo-1706325640620-7a0a11f5d91f"
  }
];

export function TeamPage() {
  return (
    <div className="bg-white py-24 sm:py-32">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl lg:max-w-4xl">
          <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">Meet our team</h2>
          <p className="mt-6 text-lg leading-8 text-gray-600">
            We're a dynamic group of individuals who are passionate about what we do and dedicated to delivering the best music recommendations for your mood.
          </p>
          
          <div className="mt-20 space-y-20">
            {teamMembers.map((member) => (
              <div key={member.name} className="relative flex flex-col gap-8 sm:flex-row">
                <div className="sm:w-1/3">
                  <img
                    src={member.image}
                    alt={member.name}
                    className="aspect-[4/5] w-full rounded-2xl object-cover"
                  />
                </div>
                <div className="sm:w-2/3 sm:pl-8">
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